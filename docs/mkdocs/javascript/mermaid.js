// Load Mermaid only on pages that contain diagrams.
let mermaidLoading;

function loadMermaid() {
  if (window.mermaid) {
    return Promise.resolve();
  }
  if (!mermaidLoading) {
    mermaidLoading = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://unpkg.com/mermaid@10/dist/mermaid.min.js";
      script.async = true;
      script.onload = () => {
        mermaid.initialize({
          startOnLoad: false,
          theme: "default",
          securityLevel: "loose",
          flowchart: {
            useMaxWidth: true,
            htmlLabels: true,
          },
        });
        resolve();
      };
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }
  return mermaidLoading;
}

document$.subscribe(() => {
  if (!document.querySelector(".mermaid")) {
    return;
  }
  loadMermaid().then(() => {
    mermaid.run({
      querySelector: ".mermaid",
      nodes: document.querySelectorAll(".mermaid"),
    });
  });
});
