// Theme Toggle Logic
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const body = document.documentElement;

// Initialize theme from storage
const savedTheme = localStorage.getItem('theme') || 'light';
body.setAttribute('data-theme', savedTheme);
updateIcon(savedTheme);

if(themeToggle) {
    themeToggle.addEventListener('click', () => {
        const current = body.getAttribute('data-theme');
        const newTheme = current === 'dark' ? 'light' : 'dark';
        body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateIcon(newTheme);
    });
}

function updateIcon(theme) {
    if(themeIcon) {
        themeIcon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    }
}

// Detection Form Logic (Only runs if elements exist)
const form = document.getElementById('detectionForm');
const spinner = document.getElementById('spinner');
const resultCard = document.getElementById('resultCard');
const scoreEl = document.getElementById('confidenceScore');
const textEl = document.getElementById('predictionText');

if(form) {
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        form.style.display = 'none';
        spinner.style.display = 'block';
        resultCard.style.display = 'none';

        // Simulate AI Request
        setTimeout(() => {
            spinner.style.display = 'none';
            
            const isReal = Math.random() > 0.5;
            const confidence = (Math.random() * (99.9 - 75) + 75).toFixed(1);

            resultCard.className = 'glass-card result-card mt-4 ' + (isReal ? 'real' : 'fake');
            scoreEl.textContent = confidence + '%';
            scoreEl.style.color = isReal ? 'var(--success-color)' : 'var(--danger-color)';
            textEl.textContent = isReal ? 'Likely REAL News' : 'Likely FAKE News';
            
            resultCard.style.display = 'block';
        }, 2500);
    });
}

// Global reset form function
window.resetForm = function() {
    if(form) {
        form.reset();
        form.style.display = 'block';
        resultCard.style.display = 'none';
    }
};

// Initialize AOS Animations
document.addEventListener('DOMContentLoaded', () => {
    if(typeof AOS !== 'undefined') {
        AOS.init({ duration: 800, once: true });
    }
});

// Navbar Scroll Effect
const mainNav = document.getElementById('mainNav');
if (mainNav) {
    window.addEventListener('scroll', () => {
        if (window.scrollY > 20) {
            mainNav.classList.add('scrolled');
        } else {
            mainNav.classList.remove('scrolled');
        }
    });
}
// Fetch Hero Stats from Backend
document.addEventListener('DOMContentLoaded', async () => {
    const baseUrls = ['http://127.0.0.1:5000', 'http://127.0.0.1:3402', 'https://tafaftire-detection-system-scui.onrender.com'];
    let fetchedData = null;
    
    for (const base of baseUrls) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);
            const response = await fetch(base + '/api/admin/analysis_history', { signal: controller.signal });
            clearTimeout(timeoutId);
            if (response.ok) {
                fetchedData = await response.json();
                break;
            }
        } catch (e) {
            console.warn('Fetch failed from', base, e);
        }
    }
    
    if (fetchedData && Array.isArray(fetchedData)) {
        const total = fetchedData.length;
        const fakes = fetchedData.filter(item => {
            return item.label && item.label.toLowerCase().includes('fake');
        }).length;
        const realNews = total - fakes;

        const totalEl = document.getElementById('hero-news-checked');
        const fakeEl = document.getElementById('hero-fake-news');
        const realEl = document.getElementById('hero-real-news');
        if (totalEl) totalEl.textContent = total;
        if (fakeEl) fakeEl.textContent = fakes;
        if (realEl) realEl.textContent = realNews;
    }

    // Remove old LocalStorage visitor tracking (no longer needed)
});
