"""Render cameras only when the policy consumes a new observation."""


class ObservationRenderEnv:
    """Keep every physics/control step while omitting unused camera frames.

    Construct the underlying LIBERO env with use_camera_obs=False. A fresh
    policy_observation draws each requested camera once and caches the images
    until the next step/reset/state change. This also works across the vector
    environment's existing get_env_attr protocol.
    """

    def __init__(self, env, cameras, resolution):
        self._env = env
        self._camera_names = ["robot0_eye_in_hand" if camera == "wrist" else camera for camera in cameras]
        self._resolution = resolution
        self._policy_observation = None
        self.rendered_observations = 0
        self.rendered_camera_frames = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kwargs):
        self._policy_observation = None
        self.rendered_observations = 0
        self.rendered_camera_frames = 0
        return self._env.reset(**kwargs)

    def set_init_state(self, state):
        self._policy_observation = None
        return self._env.set_init_state(state)

    def get_observation(self):
        return self._env.env._get_observations()

    def step(self, action):
        self._policy_observation = None
        return self._env.step(action)

    @property
    def policy_observation(self):
        if self._policy_observation is None:
            # Refresh derived geometry and proprioception at the window boundary.
            # forward() does not integrate physics or execute another action.
            self._env.sim.forward()
            self._env._update_observables(force=True)
            observation = dict(self._env.env._get_observations())
            for camera in self._camera_names:
                # Reuse robosuite's sensor for the exact image orientation and
                # camera convention, without enabling per-step camera observables.
                sensors, names = self._env.env._create_camera_sensors(
                    camera,
                    cam_w=self._resolution,
                    cam_h=self._resolution,
                    cam_d=False,
                    cam_segs=None,
                    modality="image",
                )
                observation[names[0]] = sensors[0]({})
            self._policy_observation = observation
            self.rendered_observations += 1
            self.rendered_camera_frames += len(self._camera_names)
        return self._policy_observation
