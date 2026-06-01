// Enables MathJax rendering only on pages that contain math.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

let mathJaxLoading;

function loadMathJax() {
  if (window.MathJax.typesetPromise) {
    return Promise.resolve();
  }
  if (!mathJaxLoading) {
    mathJaxLoading = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://unpkg.com/mathjax@3.2.2/es5/tex-mml-chtml.js";
      script.async = true;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }
  return mathJaxLoading;
}

document$.subscribe(() => {
  if (!document.querySelector(".arithmatex")) {
    return;
  }
  loadMathJax().then(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
});
