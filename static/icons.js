// Mathboard's line icons (docs/DESIGN.md, Icons). A plain script, loaded before app.js, so the
// board (a classic script) and the ES modules (graph/, nn/) share one set.
//
//   mathboardIcons.svg(name, { size = 16, cls = '' }) -> '<svg ...>' (an unknown name gives '')
//   mathboardIcons.names                                -> every icon name
//   mathboardIcons.hydrate(root = document)            -> fills every empty [data-icon] element
//
// Every icon is drawn on a 24 x 24 grid, stroked with currentColor (1.8 wide, round caps and
// joins), so it takes the text colour of its button. Filled marks say fill="currentColor".
// index.html's static markup uses <span data-icon="name"></span>; hydrate() runs once when this
// script loads and can be called again for markup built later.
(function () {
  'use strict';
  const F = 'fill="currentColor" stroke="none"';
  const dot = (x, y, r = 1.6) => `<circle cx="${x}" cy="${y}" r="${r}" ${F}/>`;
  const ICONS = {
    // ---------------------------------------------------------------- tabs
    board: '<path d="M4 20h4.5L19.5 9a2.1 2.1 0 0 0 0-3l-1.5-1.5a2.1 2.1 0 0 0-3 0L4 15.5V20z"/><path d="M13.5 6.5l4 4"/>',
    cube: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M4 7.5l8 4.5 8-4.5"/><path d="M12 12v9"/>',
    net: '<path d="M5 6l10.1 4.7M5 12h9.8M5 18l10.1-4.7"/><circle cx="18" cy="12" r="3.2"/>'
      + dot(5, 6, 2) + dot(5, 12, 2) + dot(5, 18, 2),

    // ---------------------------------------------------------------- editing
    plus: '<path d="M12 5v14M5 12h14"/>',
    minus: '<path d="M5 12h14"/>',
    close: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13"/>',
    trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12.5a1.5 1.5 0 0 0 1.5 1.5h7a1.5 1.5 0 0 0 1.5-1.5L18 7M9 7V4.5h6V7"/>',
    sparkle: '<path d="M12 3.5l1.9 6.6 6.6 1.9-6.6 1.9L12 20.5l-1.9-6.6L3.5 12l6.6-1.9L12 3.5z"/>',
    layout: '<rect x="4" y="5" width="4" height="14" rx="1"/><rect x="10" y="5" width="4" height="14" rx="1"/><rect x="16" y="5" width="4" height="14" rx="1"/>',
    fit: '<path d="M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15"/>',
    dice: '<rect x="4" y="4" width="16" height="16" rx="3.5"/>' + dot(8.8, 8.8) + dot(12, 12) + dot(15.2, 15.2),
    home: '<path d="M4 11l8-7 8 7"/><path d="M6 9.5V20h12V9.5"/><path d="M10 20v-5h4v5"/>',
    search: '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.6-4.6"/>',
    more: dot(6, 12, 1.7) + dot(12, 12, 1.7) + dot(18, 12, 1.7),
    pin: '<path d="M9 4h6l-1 5.5 3 3V14H7v-1.5l3-3L9 4z"/><path d="M12 14v6"/>',
    pencil: '<path d="M13 20h7"/><path d="M15.6 4.6a2.1 2.1 0 0 1 3 3L8.2 18l-4 1 1-4L15.6 4.6z"/>',   // rename, edit (board is the tab's pencil)
    settings: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',

    // ---------------------------------------------------------------- disclosure
    'chevron-down': '<path d="M6 9l6 6 6-6"/>',
    'chevron-up': '<path d="M6 15l6-6 6 6"/>',
    'chevron-left': '<path d="M15 6l-6 6 6 6"/>',
    'chevron-right': '<path d="M9 6l6 6-6 6"/>',

    // ---------------------------------------------------------------- playback
    play: '<path d="M7.5 5.2v13.6a.8.8 0 0 0 1.2.7l10.6-6.8a.8.8 0 0 0 0-1.4L8.7 4.5a.8.8 0 0 0-1.2.7z" ' + F + '/>',
    pause: '<rect x="6.5" y="5" width="3.6" height="14" rx="1" ' + F + '/><rect x="13.9" y="5" width="3.6" height="14" rx="1" ' + F + '/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="2" ' + F + '/>',
    'step-forward': '<path d="M6 6.2v11.6a.8.8 0 0 0 1.2.7l8.6-5.8a.8.8 0 0 0 0-1.4L7.2 5.5a.8.8 0 0 0-1.2.7z" ' + F + '/><path d="M18.5 5.5v13"/>',
    'step-back': '<path d="M18 6.2v11.6a.8.8 0 0 1-1.2.7l-8.6-5.8a.8.8 0 0 1 0-1.4l8.6-5.8a.8.8 0 0 1 1.2.7z" ' + F + '/><path d="M5.5 5.5v13"/>',
    reset: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3L4.5 9"/><path d="M4.5 4.5V9H9"/>',
    record: '<circle cx="12" cy="12" r="6" ' + F + '/>',

    // ---------------------------------------------------------------- Net tab
    weights: '<path d="M7.2 18h9.6"/>' + dot(5, 18, 2.2) + dot(19, 18, 2.2)
      + '<rect x="7.5" y="4.5" width="9" height="6.5" rx="1.8"/><path d="M12 11v3.5"/>',
    lens: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/>',
    layers: '<path d="M12 4l8 4-8 4-8-4 8-4z"/><path d="M4 12l8 4 8-4"/><path d="M4 16l8 4 8-4"/>',
    flow: '<rect x="3" y="6" width="6.5" height="12" rx="1.5"/><path d="M11.5 12h4M13.6 9.9l2.1 2.1-2.1 2.1"/><path d="M18 18v-5M21 18V8"/>',   // a matrix into the next-word bars
    train: '<path d="M4 4v16h16"/><path d="M7 7c1.5 6 4.5 9 13 9.5"/>',
    attention: '<path d="M5 16a7 7 0 0 1 14 0"/><path d="M12 16a3.5 3.5 0 0 1 7 0"/>' + dot(5, 17.5, 1.9) + dot(12, 17.5, 1.9) + dot(19, 17.5, 1.9),
    surface: '<path d="M3 15l5-3.5 4 2.5 5-5 4 3"/><path d="M3 19l5-3.5 4 2.5 5-5 4 3" opacity="0.5"/><path d="M3 11l5-3.5 4 2.5 5-5 4 3" opacity="0.5"/>',
    explain: '<path d="M5.5 4.5h13A1.5 1.5 0 0 1 20 6v9a1.5 1.5 0 0 1-1.5 1.5H12l-4.5 3.5v-3.5h-2A1.5 1.5 0 0 1 4 15V6a1.5 1.5 0 0 1 1.5-1.5z"/><path d="M8 9h8M8 12.5h5"/>',
    file: '<path d="M14 3.5H7.5A1.5 1.5 0 0 0 6 5v14a1.5 1.5 0 0 0 1.5 1.5h9A1.5 1.5 0 0 0 18 19V7.5l-4-4z"/><path d="M14 3.5v4h4"/>',
    audience: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8.5 20h7M12 16v4"/>',
    help: '<circle cx="12" cy="12" r="8.5"/><path d="M9.6 9.6a2.5 2.5 0 0 1 4.8.9c0 1.7-2.4 2.1-2.4 3.6"/>' + dot(12, 17, 1.2),

    // ---------------------------------------------------------------- board
    sigma: '<path d="M17.5 5H6.5l6 7-6 7h11"/>',
    draw: '<path d="M3.5 17c2.5 0 3.5-9 6.5-9s3 9 6 9 3-4 4.5-4"/>',
    eraser: '<path d="M8 20h12"/><path d="M4.9 15.6l9.2-9.2a2 2 0 0 1 2.8 0l2.7 2.7a2 2 0 0 1 0 2.8L12.5 19H8.3l-3.4-3.4z"/><path d="M9.5 11l5.5 5.5"/>',
    pointer: '<path d="M5.5 4l13 6.2-5.6 1.8-2 5.8L5.5 4z"/>',
    laser: '<circle cx="12" cy="12" r="3.2" ' + F + '/><circle cx="12" cy="12" r="7.5" opacity="0.45"/>',
    download: '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>',
    upload: '<path d="M12 15.5V4.5M7 9l5-5 5 5M5 20h14"/>',
    image: '<rect x="4" y="5" width="16" height="14" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M20 16l-4.5-4.5L6.5 19"/>',

    // ---------------------------------------------------------------- 3D tab
    link: '<path d="M10 7.5H7.5a4.5 4.5 0 0 0 0 9H10"/><path d="M14 7.5h2.5a4.5 4.5 0 0 1 0 9H14"/><path d="M8.5 12h7"/>',
    'right-angle': '<path d="M5 4.5V19h14.5"/><path d="M5 13h6v6"/>',
  };
  ICONS.edit = ICONS.pencil;

  function svg(name, { size = 16, cls = '' } = {}) {
    const body = ICONS[name];
    if (!body) return '';
    return `<svg class="ui-icon${cls ? ' ' + cls : ''}" viewBox="0 0 24 24" width="${size}" height="${size}" fill="none"`
      + ' stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
      + ` aria-hidden="true" focusable="false">${body}</svg>`;
  }

  function hydrate(root = document) {
    for (const el of root.querySelectorAll('[data-icon]')) {
      if (!el.firstElementChild) el.innerHTML = svg(el.dataset.icon, { size: Number(el.dataset.size) || 16 });
    }
  }

  window.mathboardIcons = Object.freeze({ svg, hydrate, names: Object.freeze(Object.keys(ICONS)) });
  hydrate();
})();
