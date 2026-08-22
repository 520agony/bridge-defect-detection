/**
 * DriftWall — 3D 漂移病害图片墙（登录页背景）
 * 原生 JS 实现（翻译自 React 组件），纯 DOM + CSS 3D transform + rAF。
 * 每列以不同速度/方向无限滚动，整墙跟随鼠标视差倾斜，hover 某列暂停、图片抬起。
 */
(function () {
  const container = document.getElementById('driftWall');
  if (!container) return;

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // 7 类病害样例图（每类 4 张，带标注框）
  const IMAGES = [
    { image: 'assets/samples/crack_1.jpg', title: '裂缝' },
    { image: 'assets/samples/crack_2.jpg', title: '裂缝' },
    { image: 'assets/samples/crack_3.jpg', title: '裂缝' },
    { image: 'assets/samples/crack_4.jpg', title: '裂缝' },
    { image: 'assets/samples/spalling_1.jpg', title: '剥落' },
    { image: 'assets/samples/spalling_2.jpg', title: '剥落' },
    { image: 'assets/samples/spalling_3.jpg', title: '剥落' },
    { image: 'assets/samples/spalling_4.jpg', title: '剥落' },
    { image: 'assets/samples/honeycomb_1.jpg', title: '蜂窝麻面' },
    { image: 'assets/samples/honeycomb_2.jpg', title: '蜂窝麻面' },
    { image: 'assets/samples/honeycomb_3.jpg', title: '蜂窝麻面' },
    { image: 'assets/samples/honeycomb_4.jpg', title: '蜂窝麻面' },
    { image: 'assets/samples/hole_1.jpg', title: '孔洞' },
    { image: 'assets/samples/hole_2.jpg', title: '孔洞' },
    { image: 'assets/samples/hole_3.jpg', title: '孔洞' },
    { image: 'assets/samples/hole_4.jpg', title: '孔洞' },
    { image: 'assets/samples/rebar_exposed_1.jpg', title: '露筋' },
    { image: 'assets/samples/rebar_exposed_2.jpg', title: '露筋' },
    { image: 'assets/samples/rebar_exposed_3.jpg', title: '露筋' },
    { image: 'assets/samples/rebar_exposed_4.jpg', title: '露筋' },
    { image: 'assets/samples/seepage_1.jpg', title: '渗水' },
    { image: 'assets/samples/seepage_2.jpg', title: '渗水' },
    { image: 'assets/samples/seepage_3.jpg', title: '渗水' },
    { image: 'assets/samples/seepage_4.jpg', title: '渗水' },
    { image: 'assets/samples/corrosion_1.jpg', title: '锈蚀' },
    { image: 'assets/samples/corrosion_2.jpg', title: '锈蚀' },
    { image: 'assets/samples/corrosion_3.jpg', title: '锈蚀' },
    { image: 'assets/samples/corrosion_4.jpg', title: '锈蚀' },
  ];

  const CFG = {
    columns: 8, tileWidth: 220, tileHeight: 150, gap: 16, radius: 12,
    tilt: 12, turn: -10, roll: 0, perspective: 1200, depth: 100,
    speed: 30, direction: 'up', variance: 0.4, parallax: 0.4,
    lift: 46, fade: 0.6, dim: 0.6, overlayColor: '#0a0d12'
  };

  // 设置 CSS 变量
  const setVar = (k, v) => container.style.setProperty(k, v);
  setVar('--dw-tile-w', CFG.tileWidth + 'px');
  setVar('--dw-tile-h', CFG.tileHeight + 'px');
  setVar('--dw-gap', CFG.gap + 'px');
  setVar('--dw-radius', CFG.radius + 'px');
  setVar('--dw-lift', CFG.lift + 'px');
  setVar('--dw-dim', CFG.dim);
  setVar('--dw-overlay', CFG.overlayColor);

  // 列分配
  const columnItems = Array.from({ length: CFG.columns }, () => []);
  IMAGES.forEach((item, i) => columnItems[i % CFG.columns].push(item));
  columnItems.forEach(col => { if (!col.length) col.push(IMAGES[0]); });

  // 速度：黄金比例伪随机 + 奇偶交替方向
  const dirSign = CFG.direction === 'up' ? 1 : -1;
  const columnFactor = (i, v) => { const p = ((i * 0.6180339887 + 0.35) % 1) * 2 - 1; return 1 + v * p; };
  const baseVelocities = columnItems.map((_, c) =>
    CFG.speed * columnFactor(c, CFG.variance) * dirSign * (c % 2 === 0 ? 1 : -1));

  // 构建 DOM
  const plane = document.createElement('div');
  plane.className = 'drift-wall__plane';
  const tracks = [];
  let meta = [];
  let offsets = [];
  let velocities = columnItems.map(() => 0);

  function buildMeta() {
    const unit = CFG.tileHeight + CFG.gap;
    return columnItems.map(col => {
      const copyHeight = Math.max(unit, col.length * unit);
      const copies = Math.max(2, Math.ceil((container.clientHeight * 1.6) / copyHeight) + 1);
      return { copyHeight, copies };
    });
  }

  function buildDOM() {
    plane.innerHTML = '';
    tracks.length = 0;
    meta.forEach((m, c) => {
      const col = document.createElement('div');
      col.className = 'drift-wall__col';
      const track = document.createElement('div');
      track.className = 'drift-wall__track';
      for (let copy = 0; copy < m.copies; copy++) {
        columnItems[c].forEach((item, itemIndex) => {
          const tile = document.createElement('div');
          tile.className = 'drift-wall__tile';
          tile.dataset.tileId = `${c}-${copy}-${itemIndex}`;
          tile.dataset.col = c;
          tile.setAttribute('role', 'img');
          tile.setAttribute('aria-label', item.title);
          const inner = document.createElement('span');
          inner.className = 'drift-wall__inner';
          const img = document.createElement('img');
          img.src = item.image; img.alt = item.title;
          img.loading = 'lazy'; img.decoding = 'async'; img.draggable = false;
          const ov = document.createElement('span');
          ov.className = 'drift-wall__overlay';
          ov.setAttribute('aria-hidden', 'true');
          inner.appendChild(img); inner.appendChild(ov);
          tile.appendChild(inner);
          track.appendChild(tile);
        });
      }
      col.appendChild(track);
      plane.appendChild(col);
      tracks.push(track);
    });
    container.appendChild(plane);
  }

  function rebuild() {
    meta = buildMeta();
    offsets = meta.map((m, c) => m.copyHeight * ((c * 0.37) % 1));
    velocities = columnItems.map(() => 0);
    buildDOM();
  }
  rebuild();

  // 视差 + 动画
  let pointer = { x: 0, y: 0 };
  let damped = { x: 0, y: 0 };
  let hoveredCol = -1;
  let lastTs = null;

  function applyPlaneTransform(px, py) {
    plane.style.transform =
      `translate(-50%, -50%) scale(1.18) rotateX(${CFG.tilt + py}deg) rotateY(${CFG.turn + px}deg) ` +
      `rotateZ(${CFG.roll}deg) translateZ(${-CFG.depth}px)`;
  }

  function animate(ts) {
    if (lastTs === null) lastTs = ts;
    const dt = Math.min(0.05, Math.max(0, ts - lastTs) / 1000);
    lastTs = ts;

    if (CFG.parallax > 0 && !reduced) {
      const maxTilt = CFG.parallax * 8;
      const damp = 1 - Math.exp(-dt / 0.12);
      damped.x += (pointer.x * maxTilt - damped.x) * damp;
      damped.y += (-pointer.y * maxTilt - damped.y) * damp;
    }
    applyPlaneTransform(damped.x, damped.y);

    if (!reduced) {
      for (let c = 0; c < tracks.length; c++) {
        const m = meta[c]; if (!m) continue;
        const factor = hoveredCol === c ? 0 : 1;
        const target = baseVelocities[c] * factor;
        const ease = 1 - Math.exp(-dt / (target === 0 ? 0.16 : 0.28));
        velocities[c] += (target - velocities[c]) * ease;
        let next = (offsets[c] ?? 0) + velocities[c] * dt;
        next = ((next % m.copyHeight) + m.copyHeight) % m.copyHeight;
        offsets[c] = next;
        tracks[c].style.transform = `translate3d(0, ${-next}px, 0)`;
      }
    }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  // 鼠标交互
  function setHover(col) {
    if (col === hoveredCol) return;
    hoveredCol = col;
    plane.querySelectorAll('.drift-wall__tile').forEach(t =>
      t.classList.toggle('is-active', t.dataset.col == col));
  }

  container.addEventListener('pointermove', e => {
    const rect = container.getBoundingClientRect();
    pointer = { x: (e.clientX - rect.left) / rect.width - 0.5, y: (e.clientY - rect.top) / rect.height - 0.5 };
    const tile = e.target.closest && e.target.closest('[data-tile-id]');
    if (tile) setHover(Number(tile.dataset.col)); else setHover(-1);
  });
  container.addEventListener('pointerleave', () => {
    pointer = { x: 0, y: 0 };
    setHover(-1);
  });

  // 响应式
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => rebuild());
    ro.observe(container);
  }
})();
