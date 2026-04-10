document$.subscribe(function() {
  const path = window.location.pathname;
  if (!path.includes('/Examples/') || !path.endsWith('/index.html')) return;

  const tocInner = document.querySelector('.md-sidebar--secondary .md-sidebar__inner');
  if (!tocInner || tocInner.querySelector('.marimo-open-btn')) return;

  const div = document.createElement('div');
  div.style.padding = '0.8em 0.6em 0';
  div.innerHTML = '<a href="exported.html" class="md-button md-button--primary marimo-open-btn" style="width:100%">Open Marimo notebook</a>';
  tocInner.appendChild(div);
});
