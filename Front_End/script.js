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
    const baseUrls = ['https://tafaftire-detection-system-scui.onrender.com'];
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

// Auto-scroll for mobile cards (RTL layout)
document.addEventListener('DOMContentLoaded', () => {
    const scrollContainers = document.querySelectorAll('#how-it-works .row.g-4, #fake-sources .row.g-4, #features .row.g-4, #testimonials .row.g-4');
    
    scrollContainers.forEach(container => {
        let scrollInterval;
        let isInteracting = false;

        const startAutoScroll = () => {
            stopAutoScroll();
            if (window.innerWidth <= 768) {
                scrollInterval = setInterval(() => {
                    if (!isInteracting) {
                        const maxScroll = container.scrollWidth - container.clientWidth;
                        if (Math.abs(container.scrollLeft) >= maxScroll - 10) {
                            container.scrollTo({ left: 0, behavior: 'smooth' });
                        } else {
                            container.scrollBy({ left: -container.clientWidth, behavior: 'smooth' });
                        }
                    }
                }, 3000); // Automatically slide every 3 seconds
            }
        };

        const stopAutoScroll = () => {
            if (scrollInterval) clearInterval(scrollInterval);
        };

        startAutoScroll();

        // Pause auto-scroll when user touches/interacts with the slider
        container.addEventListener('touchstart', () => { 
            isInteracting = true; 
            stopAutoScroll(); 
        }, {passive: true});
        
        container.addEventListener('touchend', () => { 
            isInteracting = false; 
            setTimeout(startAutoScroll, 3000); // Resume after 3s of no interaction
        }, {passive: true});

        window.addEventListener('resize', startAutoScroll);
    });
});

// Contact Form Submission Logic
document.addEventListener('DOMContentLoaded', () => {
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const name = document.getElementById('contactName').value.trim();
            const email = document.getElementById('contactEmail').value.trim();
            const message = document.getElementById('contactMessage').value.trim();
            const submitBtn = document.getElementById('contactSubmitBtn');
            const btnText = submitBtn.querySelector('span');
            
            // Disable button to prevent multiple submissions
            submitBtn.disabled = true;
            const originalText = btnText.innerHTML;
            btnText.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Diraya...';
            
            try {
                // Hadda waxaan dib ugu xirnay Render (Live Server-ka Internet-ka)
                const response = await fetch('https://tafaftire-detection-system-scui.onrender.com/contact', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ name, email, message })
                });
                
                if (response.ok) {
                    alert('Fariintaada si guul ah ayaa loo diray! Mahadsanid.');
                    contactForm.reset();
                } else {
                    alert('Cilad ayaa dhacday! Fadlan mar kale isku day.');
                }
            } catch (error) {
                console.error('Error submitting contact form:', error);
                alert('Cilad ayaa dhacday! Fadlan iska hubi internet-kaaga.');
            } finally {
                submitBtn.disabled = false;
                btnText.innerHTML = originalText;
            }
        });
    }
});
