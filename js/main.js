/* ═══════════════════════════════════════════
   MODE(LABS) — MAIN JAVASCRIPT (Stage 1)
   ═══════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

  // ─── REGISTER GSAP PLUGINS ───
  gsap.registerPlugin(ScrollTrigger);

  // ─── NAV: scroll behaviour ───
  const nav = document.getElementById('nav');
  const handleNavScroll = () => {
    nav.classList.toggle('scrolled', window.scrollY > 40);
  };
  window.addEventListener('scroll', handleNavScroll, { passive: true });

  // ─── HAMBURGER ───
  const hamburger = document.getElementById('hamburger');
  const mobileMenu = document.getElementById('mobileMenu');
  if (hamburger && mobileMenu) {
    hamburger.addEventListener('click', () => {
      hamburger.classList.toggle('open');
      mobileMenu.classList.toggle('open');
    });
    mobileMenu.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        hamburger.classList.remove('open');
        mobileMenu.classList.remove('open');
      });
    });
  }

  // ─── HERO ANIMATIONS (GSAP Timeline) ───
  const heroTL = gsap.timeline({ defaults: { ease: 'power3.out' } });

  heroTL
    .to('.hero-eyebrow', { opacity: 1, y: 0, duration: 0.7 }, 0.3)
    .to('.hero-title .line:nth-child(1)', { opacity: 1, y: 0, duration: 0.8 }, 0.5)
    .to('.hero-title .line:nth-child(2)', { opacity: 1, y: 0, duration: 0.8 }, 0.65)
    .to('.hero-desc',   { opacity: 1, y: 0, duration: 0.7 }, 0.8)
    .to('.hero-cta',    { opacity: 1, y: 0, duration: 0.7 }, 0.95)
    .to('.hero-badges', { opacity: 1, y: 0, duration: 0.7 }, 1.1)
    .to('.hero-visual', { opacity: 1, x: 0, duration: 0.9, ease: 'power2.out' }, 0.6);

  // ─── HERO PARALLAX: mouse move ───
  const hero = document.querySelector('.hero');
  const blob1 = document.querySelector('.blob-1');
  const blob2 = document.querySelector('.blob-2');
  const blob3 = document.querySelector('.blob-3');
  const heroContent = document.querySelector('.hero-content');
  const heroVisual = document.querySelector('.hero-visual');

  if (hero && window.innerWidth > 768) {
    hero.addEventListener('mousemove', (e) => {
      const rect = hero.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width  * 2 - 1;
      const y = (e.clientY - rect.top)  / rect.height * 2 - 1;

      gsap.to(blob1, { x: x * 50,  y: y * 35,  duration: 2.5, ease: 'power2.out' });
      gsap.to(blob2, { x: -x * 30, y: -y * 25, duration: 3,   ease: 'power2.out' });
      gsap.to(blob3, { x: x * 20,  y: y * 50,  duration: 3.5, ease: 'power2.out' });
      gsap.to(heroContent, { x: x * 8,  y: y * 5, duration: 2, ease: 'power2.out' });
      gsap.to(heroVisual,  { x: -x * 14, y: -y * 10, duration: 2, ease: 'power2.out' });
    });
  }

  // ─── HERO PARALLAX: scroll ───
  gsap.to('.hero-bg', {
    scrollTrigger: {
      trigger: '.hero', scrub: 1.5,
      start: 'top top', end: 'bottom top',
    },
    y: '40%', ease: 'none',
  });

  gsap.to('.hero-content', {
    scrollTrigger: {
      trigger: '.hero', scrub: 1.5,
      start: 'top top', end: 'bottom top',
    },
    y: '-18%', opacity: 0, ease: 'none',
  });

  gsap.to('.hero-visual', {
    scrollTrigger: {
      trigger: '.hero', scrub: 2,
      start: 'top top', end: 'bottom top',
    },
    y: '-12%', ease: 'none',
  });

  // ─── SCROLL REVEAL (generic) ───
  const reveals = document.querySelectorAll('.reveal');
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -50px 0px' });
  reveals.forEach(el => revealObserver.observe(el));

  // ─── HERO GRID PARALLAX ───
  gsap.to('.hero-grid', {
    scrollTrigger: {
      trigger: '.hero', scrub: 1,
      start: 'top top', end: 'bottom top',
    },
    backgroundPosition: '0px 80px', ease: 'none',
  });

  // ─── COUNT-UP ANIMATION (for stats, used in later stages) ───
  window.countUp = (el, target, duration = 1500) => {
    const start = performance.now();
    const update = (time) => {
      const progress = Math.min((time - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(ease * target);
      if (progress < 1) requestAnimationFrame(update);
      else el.textContent = target;
    };
    requestAnimationFrame(update);
  };

  // ─── SMOOTH ANCHOR SCROLL ───
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', e => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        const top = target.getBoundingClientRect().top + window.scrollY - 80;
        window.scrollTo({ top, behavior: 'smooth' });
      }
    });
  });

  // ─── CURSOR GLOW (desktop only) ───
  if (window.innerWidth > 1024) {
    const glow = document.createElement('div');
    glow.style.cssText = `
      position:fixed; pointer-events:none; z-index:9999;
      width:320px; height:320px; border-radius:50%;
      background: radial-gradient(circle, rgba(255,94,0,0.07) 0%, transparent 70%);
      transform: translate(-50%,-50%); transition: opacity .3s;
      left:-999px; top:-999px;
    `;
    document.body.appendChild(glow);
    document.addEventListener('mousemove', (e) => {
      glow.style.left = e.clientX + 'px';
      glow.style.top  = e.clientY + 'px';
    });
  }

});

/* ─── STAGE 2: GSAP SCROLL ANIMATIONS ─── */

// Wait for DOM
document.addEventListener('DOMContentLoaded', () => {

  // ── Stats count-up ──
  const statEls = document.querySelectorAll('[data-count]');
  statEls.forEach(el => {
    const target = parseFloat(el.dataset.count);
    const isFloat = el.dataset.count.includes('.');
    ScrollTrigger.create({
      trigger: el,
      start: 'top 85%',
      once: true,
      onEnter: () => {
        const start = performance.now();
        const dur = 1800;
        const tick = (now) => {
          const p = Math.min((now - start) / dur, 1);
          const ease = 1 - Math.pow(1 - p, 3);
          el.textContent = isFloat
            ? (ease * target).toFixed(1)
            : Math.floor(ease * target);
          if (p < 1) requestAnimationFrame(tick);
          else el.textContent = isFloat ? target.toFixed(1) : target;
        };
        requestAnimationFrame(tick);
      }
    });
  });

  // ── WHY US cards stagger ──
  gsap.from('.why-card', {
    scrollTrigger: { trigger: '.why-grid', start: 'top 80%', once: true },
    y: 40, stagger: 0.15, duration: 0.75, ease: 'power3.out',
  });

  // ── Section heads reveal ──
  gsap.utils.toArray('.section-head').forEach(el => {
    gsap.from(el, {
      scrollTrigger: { trigger: el, start: 'top 85%', once: true },
      y: 25, duration: 0.7, ease: 'power2.out',
    });
  });

  // ── Services cards ──
  gsap.from('.service-card', {
    scrollTrigger: { trigger: '.services-grid', start: 'top 80%', once: true },
    y: 40, stagger: 0.2, duration: 0.85, ease: 'power3.out',
  });

  // ── Stats strip cells ──
  gsap.from('.stat-item', {
    scrollTrigger: { trigger: '.stats-strip', start: 'top 85%', once: true },
    y: 20, stagger: 0.1, duration: 0.65, ease: 'power2.out',
  });

  // ── How it works steps ──
  gsap.from('.step', {
    scrollTrigger: { trigger: '.steps-grid', start: 'top 80%', once: true },
    y: 30, stagger: 0.2, duration: 0.75, ease: 'power3.out',
  });

  // ── Steps line fill on scroll ──
  const lineFill = document.querySelector('.steps-line-fill');
  if (lineFill) {
    gsap.to(lineFill, {
      scrollTrigger: {
        trigger: '.steps-wrap', start: 'top 75%', end: 'bottom 60%', scrub: 1,
      },
      width: '100%', ease: 'none',
    });
  }

  // ── Step numbers activate on scroll ──
  const steps = document.querySelectorAll('.step');
  steps.forEach((step, i) => {
    ScrollTrigger.create({
      trigger: step, start: 'top 75%', once: true,
      onEnter: () => setTimeout(() => step.classList.add('active'), i * 200),
    });
  });

  // ── Service cards parallax ──
  document.querySelectorAll('.service-card').forEach((card, i) => {
    gsap.to(card, {
      scrollTrigger: { trigger: card, start: 'top bottom', end: 'bottom top', scrub: 2 },
      y: i % 2 === 0 ? -18 : -10, ease: 'none',
    });
  });

});

/* ═══════════════════════════════════════════
   STAGE 3 — BA SLIDER · FILTERS · LIGHTBOX
   ═══════════════════════════════════════════ */

// ── Before/After Slider ──
function initBA(el) {
  const before = el.querySelector('.ba-before');
  const handle = el.querySelector('.ba-handle');
  if (!before || !handle) return;
  let dragging = false;

  const setPos = (clientX) => {
    const r = el.getBoundingClientRect();
    const pos = Math.max(2, Math.min(98, ((clientX - r.left) / r.width) * 100));
    handle.style.setProperty('left', pos + '%', 'important');
    before.style.setProperty('clip-path', `inset(0 ${100 - pos}% 0 0)`, 'important');
  };

  // Mouse
  handle.addEventListener('mousedown', (e) => { dragging = true; e.preventDefault(); });
  window.addEventListener('mousemove', (e) => { if (dragging) setPos(e.clientX); });
  window.addEventListener('mouseup',   () => { dragging = false; });

  // Touch
  handle.addEventListener('touchstart',  (e) => { dragging = true; }, { passive: true });
  window.addEventListener('touchmove',   (e) => { if (dragging && e.touches[0]) setPos(e.touches[0].clientX); }, { passive: true });
  window.addEventListener('touchend',    () => { dragging = false; });

  // Click on container
  el.addEventListener('click', (e) => {
    if (!e.target.closest('.ba-handle')) setPos(e.clientX);
  });
}

// Init all sliders on page load
document.querySelectorAll('[data-ba]').forEach(initBA);

// ── Portfolio Filter ──
const filterBtns  = document.querySelectorAll('.filter-btn');
const portItems   = document.querySelectorAll('.portfolio-item');

if (filterBtns.length) {
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const cat = btn.dataset.filter;

      portItems.forEach((item, i) => {
        const show = cat === 'all' || item.dataset.category === cat;
        if (show) {
          item.style.display = '';
          gsap.from(item, { opacity: 0, y: 18, duration: 0.45, delay: i * 0.05, ease: 'power2.out' });
        } else {
          item.style.display = 'none';
        }
      });
    });
  });
}

// ── Lightbox ──
const lightbox    = document.getElementById('lightbox');
const lbInner     = document.querySelector('.lb-inner');
let lbItems       = [];
let lbCurrent     = 0;

function openLightbox(items, index) {
  lbItems   = items;
  lbCurrent = index;
  renderLB();
  lightbox.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  lightbox.classList.remove('open');
  document.body.style.overflow = '';
}
function renderLB() {
  const d = lbItems[lbCurrent];
  if (!d || !lbInner) return;
  const baWrap = lbInner.querySelector('.lb-ba-wrap');
  if (baWrap) {
    baWrap.innerHTML = d.html;
    const newBA = baWrap.querySelector('[data-ba]');
    if (newBA) { initBA(newBA); }
  }
  const metaName = lbInner.querySelector('.lb-meta-name');
  const metaCat  = lbInner.querySelector('.lb-meta-cat');
  if (metaName) metaName.textContent = d.name;
  if (metaCat)  metaCat.textContent  = d.cat;
}

if (lightbox) {
  document.querySelector('.lb-close')?.addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
  document.querySelector('.lb-prev')?.addEventListener('click', () => {
    lbCurrent = (lbCurrent - 1 + lbItems.length) % lbItems.length;
    renderLB();
  });
  document.querySelector('.lb-next')?.addEventListener('click', () => {
    lbCurrent = (lbCurrent + 1) % lbItems.length;
    renderLB();
  });
  document.addEventListener('keydown', (e) => {
    if (!lightbox.classList.contains('open')) return;
    if (e.key === 'Escape')     closeLightbox();
    if (e.key === 'ArrowLeft')  { lbCurrent = (lbCurrent - 1 + lbItems.length) % lbItems.length; renderLB(); }
    if (e.key === 'ArrowRight') { lbCurrent = (lbCurrent + 1) % lbItems.length; renderLB(); }
  });
}

// Portfolio item click → lightbox
document.querySelectorAll('.portfolio-item[data-lb]').forEach((item, i) => {
  item.addEventListener('click', () => {
    const all = [...document.querySelectorAll('.portfolio-item[data-lb]')]
      .map(el => ({ name: el.dataset.name, cat: el.dataset.catLabel, html: el.querySelector('.ba-wrap')?.outerHTML || '' }));
    openLightbox(all, i);
  });
});

// ── GSAP: portfolio & models scroll animations ──
document.addEventListener('DOMContentLoaded', () => {
  if (typeof gsap === 'undefined') return;

  gsap.from('.pp-card', {
    scrollTrigger: { trigger: '.portfolio-preview-grid', start: 'top 80%', once: true },
    y: 40, stagger: 0.15, duration: 0.75, ease: 'power3.out',
  });

  gsap.from('.model-card', {
    scrollTrigger: { trigger: '.models-preview-grid', start: 'top 80%', once: true },
    y: 30, stagger: 0.12, duration: 0.7, ease: 'power3.out',
  });

  gsap.from('.model-full-card', {
    scrollTrigger: { trigger: '.models-full-grid', start: 'top 80%', once: true },
    y: 25, stagger: 0.1, duration: 0.7, ease: 'power2.out',
  });

  gsap.utils.toArray('.portfolio-item').forEach((el, i) => {
    gsap.from(el, {
      scrollTrigger: { trigger: el, start: 'top 90%', once: true },
      y: 20, duration: 0.6, delay: (i % 3) * 0.1, ease: 'power2.out',
    });
  });
});

/* ═══════════════════════════════════════════
   STAGE 4 — LOADER · SCROLL · FORM
   ═══════════════════════════════════════════ */

// ── Inject global UI (works on all pages) ──
(function injectGlobalUI() {
  // Scroll progress bar
  const sp = document.createElement('div');
  sp.className = 'scroll-progress'; sp.id = 'scrollProgress';
  document.body.prepend(sp);

  // Back to top
  const btn = document.createElement('button');
  btn.className = 'back-to-top'; btn.id = 'backToTop';
  btn.setAttribute('aria-label', 'Наверх'); btn.textContent = '↑';
  document.body.appendChild(btn);

  // Page loader
  const loader = document.createElement('div');
  loader.className = 'page-loader'; loader.id = 'pageLoader';
  loader.innerHTML = `
    <div class="loader-inner">
      <div class="loader-logo-text">
        <img src="logo.svg" alt="Mode(gen)" style="height:32px;width:auto;display:block;">
      </div>
      <div class="loader-bar"><div class="loader-fill"></div></div>
    </div>`;
  document.body.appendChild(loader);

  // Hide loader after page loads
  window.addEventListener('load', () => {
    setTimeout(() => loader.classList.add('loaded'), 650);
  });

  // Scroll progress + back-to-top
  window.addEventListener('scroll', () => {
    const scrolled = window.scrollY;
    const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
    sp.style.width = (maxScroll > 0 ? (scrolled / maxScroll) * 100 : 0) + '%';
    btn.classList.toggle('visible', scrolled > 400);
  }, { passive: true });

  btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
})();

// ── Contact form (Web3Forms) ──
(function initForm() {
  const form = document.getElementById('contact-form');
  if (!form) return;

  const submitBtn   = form.querySelector('.form-submit');
  const successWrap = document.getElementById('formSuccess');

  // Custom radio (service) toggle
  form.querySelectorAll('.ss-opt').forEach(opt => {
    opt.addEventListener('click', () => {
      form.querySelectorAll('.ss-opt').forEach(o => o.classList.remove('selected'));
      opt.classList.add('selected');
      opt.querySelector('input').checked = true;
    });
  });

  // Custom checkbox (marketplace) toggle
  form.querySelectorAll('.ms-opt').forEach(opt => {
    opt.addEventListener('click', () => {
      opt.classList.toggle('selected');
      opt.querySelector('input').checked = opt.classList.contains('selected');
    });
  });

  // Submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Validate required fields
    let valid = true;
    form.querySelectorAll('[required]').forEach(field => {
      const err = field.parentElement.querySelector('.form-err');
      if (!field.value.trim()) {
        field.classList.add('error');
        if (err) err.textContent = 'Обязательное поле';
        valid = false;
      } else {
        field.classList.remove('error');
        if (err) err.textContent = '';
      }
    });
    // Check service selected
    const serviceSelected = form.querySelector('.ss-opt.selected');
    const serviceErr = form.querySelector('.service-err');
    if (!serviceSelected) {
      if (serviceErr) serviceErr.textContent = 'Выберите услугу';
      valid = false;
    } else {
      if (serviceErr) serviceErr.textContent = '';
    }
    if (!valid) return;

    // Loading state
    submitBtn.classList.add('loading');
    submitBtn.querySelector('.btn-text').textContent = 'Отправляем...';

    try {
      const data = new FormData(form);
      const res  = await fetch('https://api.web3forms.com/submit', {
        method: 'POST', body: data,
      });
      const result = await res.json();

      if (result.success) {
        // Show success
        form.style.display = 'none';
        if (successWrap) {
          successWrap.style.display = 'flex';
          if (typeof gsap !== 'undefined') {
            gsap.from(successWrap.children, { opacity: 0, y: 20, stagger: 0.12, duration: 0.5, ease: 'power2.out' });
          }
        }
      } else {
        throw new Error(result.message || 'Ошибка');
      }
    } catch (err) {
      submitBtn.classList.remove('loading');
      submitBtn.querySelector('.btn-text').textContent = 'Отправить заявку';
      // Show error toast
      const toastEl = document.createElement('div');
      toastEl.style.cssText = 'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1A1410;border:1px solid rgba(239,68,68,.4);border-radius:10px;padding:12px 20px;font-size:13px;color:#EF4444;z-index:999;white-space:nowrap;';
      toastEl.textContent = '⚠ Ошибка отправки. Напишите нам напрямую: smimilen@gmail.com';
      document.body.appendChild(toastEl);
      setTimeout(() => toastEl.remove(), 6000);
    }
  });

  // Clear error on input
  form.querySelectorAll('.form-input, .form-textarea').forEach(el => {
    el.addEventListener('input', () => {
      el.classList.remove('error');
      const err = el.parentElement.querySelector('.form-err');
      if (err) err.textContent = '';
    });
  });
})();

/* ── Portfolio BA sliders ── */
function initPortSlider(el) {
  const before = el.querySelector('.port-before');
  const handle = el.querySelector('.port-handle');
  if (!before || !handle) return;
  let dragging = false;

  const setPos = (clientX) => {
    const r = el.getBoundingClientRect();
    const pos = Math.max(2, Math.min(98, ((clientX - r.left) / r.width) * 100));
    handle.style.setProperty('left', pos + '%', 'important');
    before.style.setProperty('clip-path', `inset(0 ${100 - pos}% 0 0)`, 'important');
  };

  handle.addEventListener('mousedown',  (e) => { dragging = true; e.preventDefault(); });
  window.addEventListener('mousemove',  (e) => { if (dragging) setPos(e.clientX); });
  window.addEventListener('mouseup',    ()  => { dragging = false; });
  handle.addEventListener('touchstart', ()  => { dragging = true; }, { passive: true });
  window.addEventListener('touchmove',  (e) => { if (dragging) setPos(e.touches[0].clientX); }, { passive: true });
  window.addEventListener('touchend',   ()  => { dragging = false; });
  el.addEventListener('click', (e) => { if (!e.target.closest('.port-handle')) setPos(e.clientX); });
}

document.querySelectorAll('.port-slider[data-port-ba]').forEach(initPortSlider);
