"""Modified from .vq_ctc_finetune.py and .ctc_finetune.py"""
import contextlib
import copy
import itertools
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER, WER_phone
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.parts.perturb import process_augmentations, RandomNoisePerturbation, AudioAugmentor
from nemo.utils import logging


class UnitCTCFinetuneModel(ASRModel):
    """Todo: should modify to remove the vector quantization"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.global_rank = 0
        self.world_size = 1
        self.local_rank = 0
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus
            self.local_rank = trainer.local_rank
        self.label_type = cfg.label_type
        assert self.label_type in ['char', 'phone', 'unit', 'bpe']
        if self.label_type == 'bpe':
            self.use_bpe = True
            assert cfg.tokenizer is not None
            from nemo.collections.asr.parts.mixins import ASRBPEMixin
            self.bpe = ASRBPEMixin()
            self.bpe._setup_tokenizer(cfg.tokenizer, register_artifact=False)
            self.tokenizer = self.bpe.tokenizer

            # Initialize a dummy vocabulary
            vocabulary = self.tokenizer.tokenizer.get_vocab()

            # Set the new vocabulary
            assert len(cfg.decoder.vocabulary) == 0
            with open_dict(cfg):
                cfg.decoder.vocabulary = ListConfig(list(vocabulary.values()))
        else:
            self.use_bpe = False
            assert cfg.tokenizer is None
        self.add_end_space = cfg.add_end_space
        self.lang = cfg.lang

        super().__init__(cfg=cfg, trainer=trainer)

        if self._cfg.encoder_type == 'spec2vec':
            from nemo.collections.asr.models.spec2vec.spec2vec_model import Spec2VecEncoder
            self.encoder = Spec2VecEncoder(self._cfg.encoder)
            encoder_param_prefix = 'spec2vec_encoder.'
        elif self._cfg.encoder_type == 'st':
            from nemo.collections.asr.models.st2vec.st2vec_model import ST2VecEncoder
            self.encoder = ST2VecEncoder(self._cfg.encoder)
            encoder_param_prefix = 'st2vec_encoder.'
        elif self._cfg.encoder_type == 'feat_st':
            from nemo.collections.asr.models.st2vec.st2vec_model import FeatST2VecEncoder
            self.encoder = FeatST2VecEncoder(self._cfg.encoder)
            encoder_param_prefix = 'st2vec_encoder.'
        else:
            assert self._cfg.encoder_type == 'wav2vec'
            from nemo.collections.asr.modules.wav2vec_encoder import Wav2VecEncoderModel
            self.encoder = Wav2VecEncoderModel(self._cfg.encoder)
            encoder_param_prefix = None
        if cfg.pretrain_chkpt_path is not None:
            self.init_encoder_from_pretrain_model(self.encoder, encoder_param_prefix, cfg.pretrain_chkpt_path)
        if self._cfg.encoder_type == 'st':
            self.encoder.remove_pretraining_modules(use_teacher_encoder=self._cfg.use_teacher_encoder)
        else:
            self.encoder.remove_pretraining_modules()

        self.decoder = self.from_config_dict(self._cfg.decoder)

        self.freeze_finetune_updates = self._cfg.freeze_finetune_updates

        self.loss = CTCLoss(
            blank_id=self.decoder.blank_idx,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        # Setup metric objects
        if self.use_bpe:
            self._wer = WERBPE(
                tokenizer=self.tokenizer,
                blank_id=self.decoder.blank_idx,
                batch_dim_index=0,
                use_cer=self._cfg.get('use_cer', False),
                ctc_decode=True,
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
                lang=self.lang,
            )
        else:
            if self.label_type in ['phone', 'unit']:
                WER_class = WER_phone
            else:
                WER_class = WER
            self._wer = WER_class(
                vocabulary=self.decoder.vocabulary,
                blank_id=self.decoder.blank_idx,
                batch_dim_index=0,
                use_cer=self._cfg.get('use_cer', False),
                ctc_decode=True,
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
                strip_end_space=self.add_end_space,
            )

    @torch.no_grad()
    def transcribe(
            self, paths2audio_files: List[str], batch_size: int = 4, logprobs=False, return_hypotheses: bool = False
    ) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                        fp.write(json.dumps(entry) + '\n')

                config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'temp_dir': tmpdir}

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    logits, logits_len, greedy_predictions = self(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device),
                        global_step=None
                    )
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            hypotheses.append(logits[idx][: logits_len[idx]])
                    else:
                        current_hypotheses = self._wer.ctc_decoder_predictions_tensor(
                            greedy_predictions, predictions_len=logits_len, return_hypotheses=return_hypotheses,
                        )

                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                        hypotheses += current_hypotheses

                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
        return hypotheses

    def change_vocabulary(self, new_vocabulary: List[str]):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        assert not self.use_bpe
        if self.decoder.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.decoder.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config['vocabulary'] = new_vocabulary
            new_decoder_config['num_classes'] = len(new_vocabulary)

            del self.decoder
            self.decoder = self.from_config_dict(new_decoder_config)
            del self.loss
            self.loss = CTCLoss(
                num_classes=self.decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            )
            if self.label_type in ['phone', 'unit']:
                WER_class = WER_phone
            else:
                WER_class = WER
            self._wer = WER_class(
                vocabulary=self.decoder.vocabulary,
                batch_dim_index=0,
                use_cer=self._cfg.get('use_cer', False),
                ctc_decode=True,
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
            )

            # Update config
            OmegaConf.set_struct(self._cfg.decoder, False)
            self._cfg.decoder = new_decoder_config
            OmegaConf.set_struct(self._cfg.decoder, True)

            logging.info(f"Changed decoder to output to {self.decoder.vocabulary} vocabulary.")

    def _setup_dataloader_from_config(self, config: Optional[Dict], noise_perturb_config):
        if noise_perturb_config is not None:
            noise_perturb = RandomNoisePerturbation(**noise_perturb_config)
            augmentor = AudioAugmentor(perturbations=[(1.0, noise_perturb)])
        else:
            augmentor = None

        shuffle = config['shuffle']

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        if self.add_end_space:
            config['parser_add_end_space'] = self.add_end_space

        if self.use_bpe:
            dataset = audio_to_text_dataset.get_bpe_dataset(config=config, tokenizer=self.tokenizer,
                                                            augmentor=augmentor)
        else:
            if self.label_type == 'char':
                dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)
            elif self.label_type in ['phone', 'unit']:
                dataset = audio_to_text_dataset.get_phone_dataset(config=config, augmentor=augmentor)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config,
                                                            noise_perturb_config=self._cfg['noise_perturb'])

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config, noise_perturb_config=None)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config, noise_perturb_config=None)

    def optim_param_groups(self):
        return [{'params': self.encoder.parameters(), 'weight_decay': 0.0},
                {'params': self.decoder.parameters()}]

    # def setup_optimization(self, optim_config: Optional[Union[DictConfig, Dict]] = None):
    #     """
    #     Prepares an optimizer from a string name and its optional config parameters.
    #
    #     Args:
    #         optim_config: A dictionary containing the following keys:
    #
    #             * "lr": mandatory key for learning rate. Will raise ValueError if not provided.
    #             * "optimizer": string name pointing to one of the available optimizers in the registry. \
    #             If not provided, defaults to "adam".
    #             * "opt_args": Optional list of strings, in the format "arg_name=arg_value". \
    #             The list of "arg_value" will be parsed and a dictionary of optimizer kwargs \
    #             will be built and supplied to instantiate the optimizer.
    #     """
    #     ## modifed by me!!
    #     # If config was not explicitly passed to us
    #     if optim_config is None:
    #         # See if internal config has `optim` namespace
    #         if self._cfg is not None and hasattr(self._cfg, 'optim'):
    #             optim_config = self._cfg.optim
    #
    #     # If config is still None, or internal config has no Optim, return without instantiation
    #     if optim_config is None:
    #         logging.info('No optimizer config provided, therefore no optimizer was created')
    #         return
    #
    #     else:
    #         # Preserve the configuration
    #         if not isinstance(optim_config, DictConfig):
    #             optim_config = OmegaConf.create(optim_config)
    #
    #         # See if internal config has `optim` namespace before preservation
    #         if self._cfg is not None and hasattr(self._cfg, 'optim'):
    #             if self._cfg.optim is None:
    #                 self._cfg.optim = copy.deepcopy(optim_config)
    #             else:
    #                 with open_dict(self._cfg.optim):
    #                     self._cfg.optim = copy.deepcopy(optim_config)
    #
    #     # Setup optimizer and scheduler
    #     if optim_config is not None and isinstance(optim_config, DictConfig):
    #         optim_config = OmegaConf.to_container(optim_config, resolve=True)
    #
    #     if 'sched' in optim_config and optim_config[
    #         'sched'] is not None and self._trainer is not None:  ## this line modified!
    #         if not isinstance(self._trainer.accumulate_grad_batches, int):
    #             raise ValueError("We do not currently support gradient acculumation that is not an integer.")
    #         if self._trainer.max_steps is None:
    #             # Store information needed to calculate max_steps
    #             optim_config['sched']['t_max_epochs'] = self._trainer.max_epochs
    #             optim_config['sched']['t_accumulate_grad_batches'] = self._trainer.accumulate_grad_batches
    #             optim_config['sched']['t_limit_train_batches'] = self._trainer.limit_train_batches
    #             if self._trainer.distributed_backend is None:
    #                 optim_config['sched']['t_num_workers'] = self._trainer.num_gpus or 1
    #             elif self._trainer.distributed_backend == "ddp_cpu":
    #                 optim_config['sched']['t_num_workers'] = self._trainer.num_processes * self._trainer.num_nodes
    #             elif self._trainer.distributed_backend == "ddp":
    #                 optim_config['sched']['t_num_workers'] = self._trainer.num_gpus * self._trainer.num_nodes
    #             else:
    #                 logging.warning(
    #                     f"The lightning trainer received accelerator: {self._trainer.distributed_backend}. We "
    #                     "recommend to use 'ddp' instead."
    #                 )
    #                 optim_config['sched']['t_num_workers'] = self._trainer.num_gpus * self._trainer.num_nodes
    #         else:
    #             optim_config['sched']['max_steps'] = self._trainer.max_steps
    #
    #     # Force into DictConfig from nested structure
    #     optim_config = OmegaConf.create(optim_config)
    #     # Get back nested dict so we its mutable
    #     optim_config = OmegaConf.to_container(optim_config, resolve=True)
    #
    #     # Extract scheduler config if inside optimizer config
    #     if 'sched' in optim_config:
    #         scheduler_config = optim_config.pop('sched')
    #     else:
    #         scheduler_config = None
    #
    #     # Check if caller provided optimizer name, default to Adam otherwise
    #     optimizer_cls = optim_config.get('_target_', None)
    #
    #     if optimizer_cls is None:
    #         # Try to get optimizer name for dynamic resolution, defaulting to Adam
    #         optimizer_name = optim_config.get('name', 'adam')
    #     else:
    #         if inspect.isclass(optimizer_cls):
    #             optimizer_name = optimizer_cls.__name__.lower()
    #         else:
    #             # resolve the class name (lowercase) from the class path if not provided
    #             optimizer_name = optimizer_cls.split(".")[-1].lower()
    #
    #     # We are guarenteed to have lr since it is required by the argparser
    #     # But maybe user forgot to pass it to this function
    #     lr = optim_config.get('lr', None)
    #
    #     # Check if caller has optimizer kwargs, default to empty dictionary
    #     if 'args' in optim_config:
    #         optimizer_args = optim_config.pop('args')
    #         optimizer_args = optim.parse_optimizer_args(optimizer_name, optimizer_args)
    #     else:
    #         optimizer_args = copy.deepcopy(optim_config)
    #
    #         # Remove extra parameters from optimizer_args nest
    #         # Assume all other parameters are to be passed into optimizer constructor
    #         optimizer_args.pop('name', None)
    #         optimizer_args.pop('cls', None)
    #         optimizer_args.pop('lr', None)
    #
    #     # Adaptive schedulers don't need `lr`
    #     if lr is not None:
    #         optimizer_args['lr'] = lr
    #
    #     # Actually instantiate the optimizer
    #     if optimizer_cls is not None:
    #         if inspect.isclass(optimizer_cls):
    #             optimizer = optimizer_cls(self.optim_param_groups(), **optimizer_args)
    #             logging.info("Optimizer config = %s", str(optimizer))
    #
    #             self._optimizer = optimizer
    #
    #         else:
    #             # Attempt class path resolution
    #             try:
    #                 optimizer_cls = OmegaConf.create({'_target_': optimizer_cls})
    #                 if lr is not None:
    #                     optimizer_config = {'lr': lr}
    #                 else:
    #                     optimizer_config = {}
    #                 optimizer_config.update(optimizer_args)
    #
    #                 optimizer_instance = hydra.utils.instantiate(
    #                     optimizer_cls, self.optim_param_groups(), **optimizer_config
    #                 )  # type: DictConfig
    #
    #                 logging.info("Optimizer config = %s", str(optimizer_instance))
    #
    #                 self._optimizer = optimizer_instance
    #
    #             except Exception as e:
    #                 logging.error(
    #                     "Could not instantiate class path - {} with kwargs {}".format(
    #                         optimizer_cls, str(optimizer_config)
    #                     )
    #                 )
    #                 raise e
    #
    #     else:
    #         optimizer = optim.get_optimizer(optimizer_name)
    #         optimizer = optimizer(self.optim_param_groups(), **optimizer_args)
    #
    #         logging.info("Optimizer config = %s", str(optimizer))
    #
    #         self._optimizer = optimizer
    #
    #     # Try to instantiate scheduler for optimizer
    #     self._scheduler = prepare_lr_scheduler(
    #         optimizer=self._optimizer, scheduler_config=scheduler_config, train_dataloader=self._train_dl
    #     )
    #
    #     # Return the optimizer with/without scheduler
    #     # This return allows multiple optimizers or schedulers to be created
    #     return self._optimizer, self._scheduler

    def forward(self, input_signal, input_signal_length, global_step):
        """
        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
        """
        ft = False if global_step is None else self.freeze_finetune_updates <= global_step
        with torch.no_grad() if not ft else contextlib.suppress():
            encoded, encoded_len = self.encoder(input_signal, input_signal_length, None, None,
                                                mask=self.training, features_only=True)

        # [B, T, D] => [B, D, T]
        encoded = encoded.transpose(1, 2)

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        max_output_len = encoded_len.max()
        if encoded.shape[2] != max_output_len:
            encoded = encoded.narrow(dim=2, start=0, length=max_output_len).contiguous()

        logits, encoded_len = self.decoder(encoder_output=encoded, lens=encoded_len, log_prob=False)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        with torch.no_grad():
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions, logits

    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions, _ = self(input_signal=signal, input_signal_length=signal_len,
                                                      global_step=self.trainer.global_step)
        loss = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        tensorboard_logs = {'train_loss': loss, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0, decode_results=None):
        signal, signal_len, transcript, transcript_len = batch
        with torch.no_grad():
            log_probs, encoded_len, predictions, logits = self(input_signal=signal, input_signal_length=signal_len,
                                                               global_step=None)
            loss = self.loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
        self._wer.update(
            predictions=predictions, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len,
            log_prediction=batch_idx < 3, decode_results=decode_results)
        wer, wer_num, wer_denom = self._wer.compute()
        return {
            'val_loss': loss,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
            'val_logprob': log_probs.cpu().numpy(),
            'val_logprob_len': encoded_len.cpu().numpy(),
            'val_logits': logits.cpu().numpy(),
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        decode_results = {}
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx, decode_results=decode_results)
        test_logs = {
            'test_loss': logs['val_loss'],
            'test_wer_num': logs['val_wer_num'],
            'test_wer_denom': logs['val_wer_denom'],
            'test_wer': logs['val_wer'],
            'test_references': decode_results['references'],
            'test_hypotheses': decode_results['hypotheses'],
            'test_logprob': logs['val_logprob'],
            'test_logprob_len': logs['val_logprob_len'],
            'test_logits': logs['val_logits'],
        }
        return test_logs

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        assert not self.use_bpe
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.decoder.vocabulary,
            'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
            'trim_silence': True,
            'shuffle': False,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config),
                                                                 noise_perturb_config=None)
        return temporary_datalayer

    @classmethod
    def init_encoder_from_pretrain_model(
            cls,
            encoder,
            encoder_param_prefix,
            checkpoint_path,
            *,
            map_location=None,
            strict: bool = True):
        try:
            cls._set_model_restore_state(is_being_restored=True)

            from pytorch_lightning.utilities.cloud_io import load as pl_load
            if map_location is not None:
                checkpoint = pl_load(checkpoint_path, map_location=map_location)
            else:
                checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

            # for past checkpoint need to add the new key
            assert cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint
            pretrain_cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]

            # give model a chance to load something
            # model.on_load_checkpoint(checkpoint)

            # load the state_dict on the model automatically
            if encoder_param_prefix is not None:
                encoder_state = {k[len(encoder_param_prefix):]: v for k, v in checkpoint['state_dict'].items() if
                                 k.startswith(encoder_param_prefix)}
            else:
                encoder_state = checkpoint['state_dict']
            encoder.load_state_dict(encoder_state, strict=strict)
        finally:
            cls._set_model_restore_state(is_being_restored=False)

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'test_loss': val_loss_mean, 'test_wer': wer_num / wer_denom}
        references = itertools.chain.from_iterable([x['test_references'] for x in outputs])
        hypotheses = itertools.chain.from_iterable([x['test_hypotheses'] for x in outputs])
        test_logprob = [x['test_logprob'] for x in outputs]
        test_logprob_len = [x['test_logprob_len'] for x in outputs]
        test_logits = [x['test_logits'] for x in outputs]
        return {'test_loss': val_loss_mean, 'log': tensorboard_logs, 'decode_results': (references, hypotheses),
                'test_logprob': test_logprob, 'test_logprob_len': test_logprob_len, 'test_logits': test_logits}
