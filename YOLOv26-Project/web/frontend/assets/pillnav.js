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
    { label: '素材演示库', sec: 'lib' },
    { label: '视频检测', sec: 'video' },
    { label: '实时摄像头', sec: 'camera' },
    { label: '融合演示', sec: 'fusion' },
    { label: '结果分析', sec: 'analysis' },
    { label: '模型对比', sec: 'compare' },
    { label: '数据集', sec: 'dataset' },
    { label: '检测历史', sec: 'history' },
    { label: '项目说明', sec: 'about' },
  ];
  const EASE = 'power3.easeOut';
  const COLORS = { base: '#ffffff', pill: '#ffffff', hover: '#120F17' };

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
  logo.innerHTML = '<span class="pill-logo-mark">🌉</span><span>桥梁病害检测</span>';
  logo.addEventListener('click', e => {
    e.preventDefault();
    if (typeof window.showSection === 'function') window.showSection('single');
    setActive('single');
  });
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
      if (w < 2 || h < 2) return; // 容器隐藏时测量为 0（登录页阶段），跳过；可见后由 ResizeObserver 自动重算
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

  // 几何布局调度：rAF 去抖 + ResizeObserver（元素从隐藏变可见时自动重算，修复首次打开错位）
  let layoutRaf = 0;
  function scheduleLayout() {
    cancelAnimationFrame(layoutRaf);
    if (document.visibilityState === 'hidden') { layout(); return; } // 后台标签 rAF 被节流：同步执行
    layoutRaf = requestAnimationFrame(layout);
  }
  window.addEventListener('resize', scheduleLayout);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) scheduleLayout(); // 后台标签中完成登录后切回前台时补一次重算
  });
  if (window.ResizeObserver) {
    try { new ResizeObserver(scheduleLayout).observe(navEl); } catch (e) { /* 忽略 */ }
  }
  if (document.fonts?.ready) document.fonts.ready.then(scheduleLayout).catch(() => {});
  scheduleLayout();

  // 暴露给外部：区块切换时同步激活态 / 从隐藏恢复时重排
  window.__pillNavSetActive = setActive;
  window.__pillNavRelayout = scheduleLayout;
})();
