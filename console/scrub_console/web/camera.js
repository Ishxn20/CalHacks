/* One owner for capture: a late permission response cannot replace the selected camera. */
(function(root) {
  "use strict";
  class DemoCameraSource {
    constructor(mediaDevices, wait = ms => new Promise(ok => setTimeout(ok, ms))) {
      this.mediaDevices = mediaDevices; this.wait = wait; this.stream = null; this.version = 0;
    }
    static stopTracks(stream) { if (stream) stream.getTracks().forEach(track => track.stop()); }
    stop() { this.version++; DemoCameraSource.stopTracks(this.stream); this.stream = null; }
    async open(deviceId = "", label = "") {
      const version = ++this.version;
      DemoCameraSource.stopTracks(this.stream); this.stream = null;
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          // Request immediately in the click handler, without resolution/facing-mode constraints.
          const stream = await this.mediaDevices.getUserMedia({video:deviceId ? {deviceId:{exact:deviceId}} : true,audio:false});
          if (version !== this.version) { DemoCameraSource.stopTracks(stream); return null; }
          this.stream = stream; return stream;
        } catch (e) {
          if (version !== this.version) return null;
          if (attempt) throw e;
          if (["NotReadableError", "AbortError"].includes(e.name)) {
            await this.wait(600);  // allow the previous Continuity Camera session to release
          } else if (["NotFoundError", "OverconstrainedError"].includes(e.name) && deviceId && label) {
            const cams = await this.mediaDevices.enumerateDevices();
            const match = cams.find(d => d.kind === "videoinput" && d.label === label);
            if (!match || match.deviceId === deviceId) throw e;
            deviceId = match.deviceId;
          } else throw e;
          if (version !== this.version) return null;
        }
      }
    }
  }
  root.DemoCameraSource = DemoCameraSource;
  if (typeof module !== "undefined" && module.exports) module.exports = DemoCameraSource;
})(typeof window === "undefined" ? globalThis : window);
