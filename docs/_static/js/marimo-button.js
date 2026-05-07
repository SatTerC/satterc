document$.subscribe(function() {
  const path = window.location.pathname;
  if (!path.includes('/Examples/')) return;

  const tocInner = document.querySelector('.md-sidebar--secondary .md-sidebar__inner');
  if (!tocInner || tocInner.querySelector('.marimo-open-btn')) return;

  const filename = path.split('/').pop().replace('.html', '-notebook.html');
  const div = document.createElement('div');
  div.style.padding = '0.8em 0.6em 1.6em';
  const icon = '🚀';
  div.innerHTML = '<a href="' + filename + '" class="md-button md-button--primary marimo-open-btn" style="width:100%;font-size:.7rem;font-weight:700">' + icon + ' Open Marimo notebook</a>';
  tocInner.prepend(div);
});

