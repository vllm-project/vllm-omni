// Response-tagged audio and drain markers preserve boundaries even when the
// next response arrives before the previous one has finished playing.
class FullDuplexPcmPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.played = new Map();
    this.started = new Set();
    this.waitFrames = 0;
    this.gaps = new Map();
    this.pendingGaps = new Map();
    this.lastResponseId = null;
    this.port.onmessage = ({data: m}) => {
      if (m.type === 'audit') { this.audit = true; return; }
      if (m.type === 'clear') {
        this.queue = []; this.played.clear(); this.started.clear(); this.waitFrames = 0;
        this.gaps.clear(); this.pendingGaps.clear(); this.lastResponseId = null;
      } else if (m.type === 'cancel') {
        this.queue = this.queue.filter(part => part.id !== m.responseId);
        this.played.delete(m.responseId); this.started.delete(m.responseId);
        this.gaps.delete(m.responseId); this.pendingGaps.delete(m.responseId);
        if (this.audit) this.port.postMessage({type:"audit-cancel",responseId:m.responseId});
      } else if (m.type === 'audio' && m.pcm?.length) {
        if (!this.started.size && !this.queue.length) this.waitFrames = Math.round(sampleRate * 0.2);
        this.queue.push({pcm:m.pcm, id:m.responseId, offset:0});
      } else if (m.type === 'drain') {
        this.queue.push({drain:true,id:m.responseId});
        this.waitFrames = 0;
      }
    };
  }
  process(_inputs, outputs) {
    const output = outputs[0][0]; output.fill(0);
    if (this.waitFrames > 0) { this.waitFrames -= output.length; return true; }
    let cursor = 0;
    while (this.queue.length && cursor < output.length) {
      const part = this.queue[0];
      if (part.drain) {
        this.queue.shift();
        this.port.postMessage({type:'playback-drained',responseId:part.id,playedMs:Math.round((this.played.get(part.id)||0)*1000/sampleRate),underrunMs:Math.round((this.gaps.get(part.id)||0)*1000/sampleRate)});
        this.played.delete(part.id); this.started.delete(part.id);
        this.gaps.delete(part.id); this.pendingGaps.delete(part.id);
        continue;
      }
      if (!this.started.has(part.id)) {
        this.started.add(part.id);
        this.port.postMessage({type:'playback-started',responseId:part.id});
      }
      // Commit a gap only when this reply resumes. Silence after its final
      // audio while awaiting the drain marker is not an internal underrun.
      const gap = this.pendingGaps.get(part.id) || 0;
      if (gap) {
        this.gaps.set(part.id, (this.gaps.get(part.id) || 0) + gap);
        this.pendingGaps.delete(part.id);
      }
      this.lastResponseId = part.id;
      const count = Math.min(output.length-cursor,part.pcm.length-part.offset);
      for(let i=0;i<count;i++) output[cursor+i] = part.pcm[part.offset+i] / 32768;
      if (this.audit) this.port.postMessage({type:'audit-render',responseId:part.id,pcm:output.slice(cursor,cursor+count)});
      cursor += count; part.offset += count;
      this.played.set(part.id,(this.played.get(part.id)||0)+count);
      if(part.offset === part.pcm.length) this.queue.shift();
    }
    if (cursor < output.length && this.started.has(this.lastResponseId)) {
      const id = this.lastResponseId;
      this.pendingGaps.set(id, (this.pendingGaps.get(id) || 0) + output.length - cursor);
    }
    this.visualTicks = (this.visualTicks || 0) + 1;
    if (this.visualTicks % 12 === 0) {
      let peak=0; for (const sample of output) peak=Math.max(peak,Math.abs(sample));
      this.port.postMessage({type:'visual-level',level:Math.min(1,peak*4)});
    }
    return true;
  }
}
registerProcessor('fullduplex-pcm-playback',FullDuplexPcmPlayback);
