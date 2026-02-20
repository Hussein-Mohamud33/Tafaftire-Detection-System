document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------
    // 0. CONFIG
    // ----------------------------
    const API_BASE_URL = 'http://127.0.0.1:3402';

    // Buttons
    const predictBtn = document.getElementById("predictBtn");
    const submitBtn = document.querySelector('.submit-btn');

    // ----------------------------
    // HEALTH CHECK
    // ----------------------------
    fetch(`${API_BASE_URL}/`)
        .then(res => res.json())
        .then(data => {
            if (data.status === "OK") {
                console.log("✅ Flask server online");
                if (predictBtn) predictBtn.disabled = false;
                if (submitBtn) submitBtn.disabled = false;
            }
        })
        .catch(err => {
            console.warn("⚠️ Server-ka Fake News Detection ma shaqeynayo.");
        });

    // ----------------------------
    // 1. SPA Navigation
    // ----------------------------
    const allInternalLinks = document.querySelectorAll('a[href^="#"]');
    const mainNavLinks = document.querySelectorAll('.nav-links a');
    const sections = document.querySelectorAll('section');
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-links');

    function showSection(sectionId) {
        if (!sectionId) return;
        sections.forEach(sec => sec.style.display = 'none');
        const target = document.getElementById(sectionId);
        if (target) {
            target.style.display = 'block';
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        mainNavLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + sectionId) link.classList.add('active');
        });
        if (navMenu) navMenu.classList.remove('active');
    }

    allInternalLinks.forEach(link => link.addEventListener('click', e => {
        const href = link.getAttribute('href');
        if (href.startsWith('#')) {
            e.preventDefault();
            const sectionId = href.substring(1);
            showSection(sectionId);
        }
    }));

    if (hamburger) hamburger.addEventListener('click', () => navMenu.classList.toggle('active'));

    const hash = window.location.hash.substring(1);
    showSection(hash && document.getElementById(hash) ? hash : 'home');

    // ----------------------------
    // 2. Hero Slider Logic (Automatic Only)
    // ----------------------------
    const slides = document.querySelectorAll('.slide');
    let currentSlide = 0;

    function showSlide(index) {
        slides.forEach((slide, i) => {
            slide.style.opacity = i === index ? '1' : '0';
            slide.style.transition = 'opacity 1s ease-in-out';
        });
    }

    function nextSlide() {
        currentSlide = (currentSlide + 1) % slides.length;
        showSlide(currentSlide);
    }

    if (slides.length > 0) {
        slides.forEach(s => s.style.animation = 'none');
        showSlide(0);
        setInterval(nextSlide, 5000);
    }

    // ----------------------------
    // 2.5 VALIDATION HELPERS & UI
    // ----------------------------
    const errorDiv = document.getElementById("errorMessage");
    const newsText = document.getElementById("newsText");
    const newsURL = document.getElementById("newsURL");
    let errorTimeout = null;

    function showError(msg, inputId) {
        if (errorDiv) {
            // Clear any existing timeout
            if (errorTimeout) clearTimeout(errorTimeout);

            errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${msg}`;

            // Clear the offending input
            const input = document.getElementById(inputId);
            if (input) input.value = "";

            // Auto-hide error after 4 seconds
            errorTimeout = setTimeout(() => {
                errorDiv.innerHTML = "";
            }, 4000);
        }
    }

    function clearError() {
        if (errorDiv) errorDiv.innerText = "";
    }

    // Clear error when user starts typing
    [newsText, newsURL].forEach(input => {
        if (input) input.addEventListener('input', clearError);
    });

    function isURL(text) {
        const urlPattern = /^(https?:\/\/)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$/i;
        const simpleDomainPattern = /[a-z0-9.-]+\.(com|net|org|io|gov|edu|info|so|me)/i;
        return urlPattern.test(text.trim()) || simpleDomainPattern.test(text.trim());
    }

    function containsLink(text) {
        const linkPattern = /(https?:\/\/[^\s]+|www\.[^\s]+|[a-z0-9.-]+\.(com|net|org|io|info|gov|edu|so|me))/i;
        return linkPattern.test(text);
    }

    function isGibberish(text) {
        if (text.length < 10) return false; // Too short to judge fairly
        const words = text.split(/\s+/);

        // 1. Check for extreme word length (no spaces for a long time)
        for (let word of words) {
            if (word.length > 35) return true;
        }

        // 2. Check vowel-to-consonant ratio (heuristic)
        const totalChars = text.replace(/\s/g, "").length;
        const vowels = text.match(/[aeiou]/gi) || [];
        if (totalChars > 20 && (vowels.length / totalChars) < 0.12) return true;

        // 3. Check for repetitive character patterns (e.g., "aaaaa" or "asdfasdf")
        if (/(.)\1{4,}/i.test(text)) return true;

        return false;
    }

    // ----------------------------
    // 3. Fake News Prediction
    // ----------------------------
    const resultDiv = document.getElementById("result");
    const confidenceDiv = document.getElementById("confidence");
    const textInput = document.getElementById("textInput");
    const urlInput = document.getElementById("urlInput");

    document.querySelectorAll('input[name="inputType"]').forEach(radio => {
        radio.addEventListener('change', () => {
            if (radio.value === "text") { textInput.classList.remove("hidden"); urlInput.classList.add("hidden"); }
            else { textInput.classList.add("hidden"); urlInput.classList.remove("hidden"); }
        });
    });

    if (predictBtn) predictBtn.addEventListener('click', () => {
        const selected = document.querySelector('input[name="inputType"]:checked');
        const inputType = selected.value;
        let data = "";
        if (inputType === "text") {
            data = newsText.value.trim();
            if (data.length < 20) { showError("Fadlan geli qoraal kugu filan (ugu yaraan 20 xaraf).", "newsText"); return; }

            // Validation 1: No Links in Text Mode
            if (containsLink(data)) {
                showError("NIDAMKA: Text mode-ka laguma ogola Links. Fadlan u bedel 'URL' mode.", "newsText");
                return;
            }

            // Validation 3: Reject Gibberish
            if (isGibberish(data)) {
                showError("KHALAD: Qoraalkan ma ahan mid la aqrin karo.", "newsText");
                return;
            }
        } else {
            data = newsURL.value.trim();
            if (!data) { showError("Fadlan geli URL.", "newsURL"); return; }

            // Validation 2: Only URLs in URL Mode
            if (!isURL(data)) {
                showError("KHALAD: Fadlan geli URL sax ah.", "newsURL");
                return;
            }
        }

        resultDiv.innerText = "⏳ Analyzing...";
        fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type: inputType, data: data })
        }).then(res => res.json()).then(res => {
            if (res.error) { resultDiv.innerText = "❌ " + res.error; }
            else {
                const isReal = res.prediction.includes("REAL");
                resultDiv.innerText = isReal ? "WAR RUN AH" : "WAR BEEN AH";
                resultDiv.style.color = isReal ? "#2ecc71" : "#e74c3c";
                confidenceDiv.innerText = "Kalsoonida: " + res.confidence;
            }
        }).catch(() => { resultDiv.innerText = "❌ Connection Error"; });
    });

    const refreshBtn = document.getElementById("refreshBtn");
    if (refreshBtn) refreshBtn.addEventListener('click', () => {
        document.getElementById("newsText").value = "";
        document.getElementById("newsURL").value = "";
        resultDiv.innerText = "";
        confidenceDiv.innerText = "";
    });

    // ----------------------------
    // 4. Contact Form Handling
    // ----------------------------
    if (submitBtn) {
        submitBtn.addEventListener('click', () => {
            const name = document.getElementById("contactName").value.trim();
            const email = document.getElementById("contactEmail").value.trim();
            const message = document.getElementById("contactMessage").value.trim();

            if (!name || !email || !message) {
                alert("Fadlan buuxi dhamaan meelaha banaan.");
                return;
            }

            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';

            fetch(`${API_BASE_URL}/contact`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, email, message })
            })
                .then(res => res.json())
                .then(res => {
                    if (res.status === "Success") {
                        alert(res.message);
                        document.getElementById("contactName").value = "";
                        document.getElementById("contactEmail").value = "";
                        document.getElementById("contactMessage").value = "";
                    } else {
                        alert("Khalad: " + (res.error || "Lama soo diri karo fariinta."));
                    }
                })
                .catch(() => {
                    alert("Khalad: Connection Error.");
                })
                .finally(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send Message';
                });
        });
    }
});
