/**
 * PillNav — 药丸导航栏（原生 JS 翻译，gsap CDN）
 * 效果：hover 圆形从底部膨胀填充药丸 + 文字双层滑动 + logo 旋转 + 移动端汉堡菜单
 */
(function () {
  const navEl = document.getElementById('pillNav');
  if (!navEl) return;

  const ITEMS = [
    { label: '智能检测', sec: 'single' },
    { label: '批量检测', sec: 'detect' },
    { label: '结果分析', sec: 'analysis' },
    { label: '数据集', sec: 'dataset' },
    { label: '检测历史', sec: 'history' },
    { label: '项目说明', sec: 'about' },
  ];
  const EASE = 'power3.easeOut';
  const COLORS = { base: '#cbd5e1', pill: '#2dd4bf', hover: '#07110e' };

  const circles = [];
  const timelines = [];
  const activeTweens = [];

  // 构建 DOM
  navEl.style.setProperty('--base', COLORS.base);
  navEl.style.setProperty('--pill-bg', COLORS.pill);
  navEl.style.setProperty('--hover-text', COLORS.hover);

  const logo = document.createElement('a');
  logo.className = 'pill-logo';
  logo.href = '#';
  logo.setAttribute('aria-label', '首页');
  logo.innerHTML = '<span class="pill-logo-mark">🌉</span>';
  navEl.appendChild(logo);

  const list = document.createElement('ul');
  list.className = 'pill-list';
  ITEMS.forEach((item, i) => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '#' + item.sec;
    a.className = 'pill';
    a.setAttribute('aria-label', item.label);
    a.dataset.sec = item.sec;

    const circle = document.createElement('span');
    circle.className = 'hover-circle';
    circle.setAttribute('aria-hidden', 'true');
    const stack = document.createElement('span');
    stack.className = 'label-stack';
    const label = document.createElement('span');
    label.className = 'pill-label';
    label.textContent = item.label;
    const labelHover = document.createElement('span');
    labelHover.className = 'pill-label-hover';
    labelHover.setAttribute('aria-hidden', 'true');
    labelHover.textContent = item.label;

    stack.appendChild(label);
    stack.appendChild(labelHover);
    a.appendChild(circle);
    a.appendChild(stack);

    a.addEventListener('click', e => {
      e.preventDefault();
      if (typeof window.showSection === 'function') window.showSection(item.sec);
      setActive(item.sec);
    });
    a.addEventListener('mouseenter', () => handleEnter(i));
    a.addEventListener('mouseleave', () => handleLeave(i));

    li.appendChild(a);
    list.appendChild(li);
    circles.push(circle);
  });
  navEl.appendChild(list);

  // logo 旋转
  const logoMark = logo.querySelector('.pill-logo-mark');
  let logoTween = null;
  logo.addEventListener('mouseenter', () => {
    if (typeof gsap === 'undefined') return;
    logoTween?.kill();
    gsap.set(logoMark, { rotate: 0 });
    logoTween = gsap.to(logoMark, { rotate: 360, duration: 0.5, ease: EASE, overwrite: 'auto' });
  });

  // 几何布局：计算每个药丸的 hover 圆
  function layout() {
    circles.forEach((circle, i) => {
      const pill = circle.parentElement;
      if (!pill) return;
      const rect = pill.getBoundingClientRect();
      const w = rect.width, h = rect.height;
      const R = ((w * w) / 4 + h * h) / (2 * h);
      const D = Math.ceil(2 * R) + 2;
      const delta = Math.ceil(R - Math.sqrt(Math.max(0, R * R - (w * w) / 4))) + 1;
      const originY = D - delta;

      circle.style.width = D + 'px';
      circle.style.height = D + 'px';
      circle.style.bottom = -delta + 'px';

      if (typeof gsap !== 'undefined') {
        gsap.set(circle, { xPercent: -50, scale: 0, transformOrigin: `50% ${originY}px` });
        const label = pill.querySelector('.pill-label');
        const white = pill.querySelector('.pill-label-hover');
        if (label) gsap.set(label, { y: 0 });
        if (white) gsap.set(white, { y: h + 12, opacity: 0 });

        timelines[i]?.kill();
        const tl = gsap.timeline({ paused: true });
        tl.to(circle, { scale: 1.2, xPercent: -50, duration: 2, ease: EASE, overwrite: 'auto' }, 0);
        if (label) tl.to(label, { y: -(h + 8), duration: 2, ease: EASE, overwrite: 'auto' }, 0);
        if (white) {
          gsap.set(white, { y: Math.ceil(h + 100), opacity: 0 });
          tl.to(white, { y: 0, opacity: 1, duration: 2, ease: EASE, overwrite: 'auto' }, 0);
        }
        timelines[i] = tl;
      }
    });
  }

  function handleEnter(i) {
    if (typeof gsap === 'undefined') return;
    const tl = timelines[i];
    if (!tl) return;
    activeTweens[i]?.kill();
    activeTweens[i] = tl.tweenTo(tl.duration(), { duration: 0.3, ease: EASE, overwrite: 'auto' });
  }
  function handleLeave(i) {
    if (typeof gsap === 'undefined') return;
    const tl = timelines[i];
    if (!tl) return;
    activeTweens[i]?.kill();
    activeTweens[i] = tl.tweenTo(0, { duration: 0.2, ease: EASE, overwrite: 'auto' });
  }

  function setActive(sec) {
    navEl.querySelectorAll('.pill').forEach(a =>
      a.classList.toggle('is-active', a.dataset.sec === sec));
  }

  window.addEventListener('resize', layout);
  if (document.fonts?.ready) document.fonts.ready.then(layout).catch(() => {});
  layout();

  // 暴露 setActive 给外部（showSection 时同步激活态）
  window.__pillNavSetActive = setActive;
})();
