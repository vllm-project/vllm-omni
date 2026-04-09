// Initialize Mermaid for diagram rendering
mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
  securityLevel: 'loose',
  flowchart: {
    useMaxWidth: true,
    htmlLabels: true
  }
});

// Render Mermaid diagrams when page content is ready
document$.subscribe(() => {
  const mermaidElements = document.querySelectorAll('.mermaid');
  if (mermaidElements.length > 0) {
    mermaid.run({
      querySelector: '.mermaid',
      nodes: mermaidElements
    });
  }
});
