const API_BASE_URL = "https://tafaftire-detection-system-scui.onrender.com";

window.isAnalyzing = false;

window.addEventListener('beforeunload', (e) => {
    if (window.isAnalyzing) {
        e.preventDefault();
        e.returnValue = 'Analysis in progress...';
        return 'Analysis in progress...';
    }
});

// ----------------------------
// 0. QUICK PING (WAKE UP RENDER)
// ----------------------------
// We fire this immediately at the top of the script to wake up Render as fast as possible.
let isServerOnline = false;
const checkServerStatus = () => {
    fetch(`${API_BASE_URL}/api/health`)
        .then(res => res.json())
        .then(data => {
            if (data.status === "OK") {
                isServerOnline = true;
                console.log("🚀 Render Server Wakeup Successful");
                const analyzeBtn = document.getElementById("analyzeBtn");
                if (analyzeBtn) analyzeBtn.disabled = false;
                
                // If preloader is still visible, trigger its removal
                const preloader = document.getElementById('preloader');
                if (preloader && !preloader.classList.contains('fade-out')) {
                    preloader.classList.add('fade-out');
                    setTimeout(() => { preloader.remove(); }, 800);
                }
            }
        })
        .catch(err => {
            console.warn("Waiting for server spin-up...");
            // Retry every 3 seconds if not online yet
            if (!isServerOnline) setTimeout(checkServerStatus, 3000);
        });
};

checkServerStatus();

// Keep-Alive every 5 mins
setInterval(() => {
    fetch(`${API_BASE_URL}/api/health`).catch(() => {});
}, 300000);

// ----------------------------
// INITIAL RENDER OPTIMIZATION
// ----------------------------
const TRUSTED_DOMAINS = ["bbc.com", "voasomali.com", "goobjoog.com", "garoweonline.com", "sntv.so", "sonna.so", "aljazeera.com", "reuters.com", "hiiraan.com", "radiomuqdisho.net", "caasimada.net"];
const FAKE_PATTERNS = ["mucjiso", "halkan riix", "waad yaabaysaa", "subxaanallaah", "nin yaaban", "abaalmarin", "guul iyo lacag", "gift"];

const ANALYSIS_CACHE = new Map();

function displayFinalResults(aiRes, fcRes, claimsRes) {
    const aiResult = document.getElementById("aiResult");
    const fcResult = document.getElementById("fcResult");
    const aiConfidence = document.getElementById("aiConfidence");
    const fcConfidence = document.getElementById("fcConfidence");
    const unifiedVerdict = document.getElementById("unifiedVerdict");
    const unifiedConfidence = document.getElementById("unifiedConfidence");
    const winnerSource = document.getElementById("winnerSource");
    const unifiedVerdictCard = document.getElementById("unifiedVerdictCard");
    const aiResultCard = document.getElementById("aiResultCard");
    const fcResultCard = document.getElementById("fcResultCard");

    // AI Display
    const aiSuccess = aiRes && !aiRes.error;
    let aiConfVal = 0;
    if (aiSuccess) {
        const aiStatus = (aiRes.prediction || "unverified").toLowerCase();
        aiResult.innerText = aiStatus.includes("real") ? "Real News" : (aiStatus.includes("fake") ? "Fake News" : "Unverified");
        aiResult.className = "prediction-small " + (aiStatus.includes("real") ? "real" : (aiStatus.includes("fake") ? "fake" : ""));
        aiConfidence.innerText = `Confidence: ${aiRes.confidence}`;
        aiConfVal = parseFloat(aiRes.confidence.replace('%', '')) || 0;
        aiResultCard.classList.remove("hidden");
    }

    // Fact Check Display
    const fcSuccess = fcRes && !fcRes.error;
    let fcConfVal = 0;
    if (fcSuccess) {
        const fcRating = (fcRes.rating || "unverified").toLowerCase();
        fcResult.innerText = fcRating.includes("trusted") ? "Trusted News" : (fcRating.includes("suspicious") ? "Suspicious" : "Unverified");
        fcResult.className = "prediction-small " + (fcRating.includes("trusted") ? "real" : (fcRating.includes("suspicious") ? "fake" : ""));
        fcConfidence.innerText = `Web Score: ${fcRes.confidence || '0%'}`;
        fcConfVal = parseFloat((fcRes.confidence || '0').replace('%', '')) || 0;
        fcResultCard.classList.remove("hidden");
    }

    // Unified Result
    if (aiSuccess || fcSuccess) {
        let winningLabel = aiResult.innerText;
        let winningConfidence = aiRes.confidence;
        
        if (fcConfVal > (aiConfVal + 10) || (aiResult.innerText === "Unverified" && fcSuccess)) {
            winningLabel = fcResult.innerText;
            winningConfidence = fcRes.confidence;
        }

        unifiedVerdict.innerText = winningLabel;
        unifiedVerdict.className = "prediction-main " + (winningLabel.includes("Real") || winningLabel.includes("Trusted") ? "real" : "fake");
        unifiedConfidence.innerText = `Final Verdict: ${winningConfidence}`;
        if (winnerSource) winnerSource.innerText = "Deep AI Engine Verified";
        unifiedVerdictCard.classList.remove("hidden");
    }
}

function quickHeuristicCheck(text, url = null) {
    let score = 50; // Neutral start
    let reasons = [];
    const textLower = text.toLowerCase();

    if (url) {
        if (TRUSTED_DOMAINS.some(d => url.toLowerCase().includes(d))) {
            return { label: "Trusted News", confidence: "95%", source: "Verified Source" };
        }
    }

    if (FAKE_PATTERNS.some(p => textLower.includes(p))) score -= 30;
    if (text.length > 500) score += 10;
    if ((text.match(/[A-Z]/g) || []).length / text.length > 0.3) score -= 15;

    if (score > 60) return { label: "Likely Real", confidence: "70%", source: "Quick Pattern Check" };
    if (score < 40) return { label: "Suspicious", confidence: "75%", source: "Quick Pattern Check" };
    return { label: "Processing...", confidence: "--", source: "Analyzing..." };
}

window.addEventListener('load', () => {
    const preloader = document.getElementById('preloader');
    const loadText = preloader ? preloader.querySelector('.loader-text') : null;
    
    // Safety Timeout: Don't keep user stuck for more than 15s even if server is slow
    const safetyTimeout = setTimeout(() => {
        if (preloader && !preloader.classList.contains('fade-out')) {
            if (loadText) loadText.innerText = "STARTING...";
            preloader.classList.add('fade-out');
            setTimeout(() => { preloader.remove(); }, 800);
        }
    }, 15000);

    if (preloader) {
        // Update text to show we are waiting for the backend
        if (loadText) loadText.innerText = "CONNECTING TO SECURE SERVERS...";
        
        // The checkServerStatus function (defined at the top) handles removal
        // but we add a small check here too in case it was already online
        if (isServerOnline) {
            clearTimeout(safetyTimeout);
            preloader.classList.add('fade-out');
            setTimeout(() => { preloader.remove(); }, 800);
        }
    }
});

document.addEventListener('DOMContentLoaded', () => {

    // Global state to hold the current result before saving to history
    let lastAnalysisResult = null;

    // Buttons
    const analyzeBtn = document.getElementById("analyzeBtn");
    const refreshBtn = document.getElementById("refreshBtn");
    const submitBtn = document.querySelector('.submit-btn');

    // ----------------------------
    // HEALTH CHECK (Already handled by Quick Ping)
    // ----------------------------
    if (isServerOnline && analyzeBtn) analyzeBtn.disabled = false;


    async function safeJson(response) {
        const text = await response.text();
        try {
            return JSON.parse(text);
        } catch (e) {
            console.error("JSON Parse Error:", text);
            throw new Error(`Server did not return JSON (HTML may be returned). Check API.`);
        }

    }

    // ----------------------------
    // 1. SPA Navigation (Static - No Scroll)
    // ----------------------------
    const allInternalLinks = document.querySelectorAll('a[href^="#"]');
    const mainNavLinks = document.querySelectorAll('.nav-links a');
    const sections = document.querySelectorAll('section');
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-links');

    function showSection(sectionId) {
        if (!sectionId) return;
        sections.forEach(sec => {
            sec.style.display = 'none';
        });
        const target = document.getElementById(sectionId);
        if (target) {
            target.style.display = 'block';
            // Scroll completely disabled as per user request
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
            if (sectionId === 'admin') {
                toggleAdminModal(true);
            } else {
                showSection(sectionId);
                // Update URL without jump
                history.replaceState(null, null, '#' + sectionId);
            }
        }
    }));

    if (hamburger) hamburger.addEventListener('click', () => navMenu.classList.toggle('active'));

    // Initialization logic moved to the bottom of DOMContentLoaded

    // ----------------------------
    // 2. Hero Slider Logic (Static - No Auto Scroll)
    // ----------------------------
    const slides = document.querySelectorAll('.slide');
    let currentSlide = 0;

    function showSlide(index) {
        slides.forEach((slide, i) => {
            slide.style.opacity = i === index ? '1' : '0';
            slide.style.transition = 'none'; // No smooth transition to avoid "scroll" feel
        });
    }

    // Auto-slide removed as per user request (no transition/scroll feel)
    if (slides.length > 0) {
        showSlide(0);
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

            const input = document.getElementById(inputId);
            if (input) {
                // If nested in premium wrapper, highlight the wrapper itself
                const highlightTarget = input.closest('.premium-input-wrapper') || input;
                highlightTarget.classList.add("invalid");
                setTimeout(() => { if (highlightTarget) highlightTarget.classList.remove("invalid"); }, 4000);
            }

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
        text = text.trim();
        // Consistently match the backend logic for extremely permissive URL detection
        const urlPattern = /^((https?:\/\/|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+|([a-z0-9-]+\.)+[a-z]{2,10})([/?#].*)?$/i;
        return urlPattern.test(text);
    }

    function containsLink(text) {
        // Matches http/www OR any string that looks like domain.tld (supports subdomains like news.somali.so)
        const linkPattern = /(https?:\/\/\S+|www\.\S+|([a-z0-9-]+\.)+[a-z]{2,10}(\/\S*)?)/i;
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
    // 3. Consolidated Analysis UI Elements
    // ----------------------------
    const resultContainer = document.getElementById("unifiedResultContainer");
    const aiResultCard = document.getElementById("aiResultCard");
    const fcResultCard = document.getElementById("fcResultCard");
    const aiResult = document.getElementById("aiResult");
    const fcResult = document.getElementById("fcResult");
    const aiConfidence = document.getElementById("aiConfidence");
    const fcConfidence = document.getElementById("fcConfidence");
    const fcReasons = document.getElementById("fcReasons");
    const factCheckBtn = document.getElementById("factCheckBtn");

    // New Winner Elements
    const unifiedVerdictCard = document.getElementById("unifiedVerdictCard");
    const unifiedVerdict = document.getElementById("unifiedVerdict");
    const unifiedConfidence = document.getElementById("unifiedConfidence");
    const winnerSource = document.getElementById("winnerSource");


    const textInput = document.getElementById("textInput");
    const urlInput = document.getElementById("urlInput");


    document.querySelectorAll('input[name="inputType"]').forEach(radio => {
        radio.addEventListener('change', () => {
            if (radio.value === "text") { textInput.classList.remove("hidden"); urlInput.classList.add("hidden"); }
            else { textInput.classList.add("hidden"); urlInput.classList.remove("hidden"); }
        });
    });

    async function performDeepAnalysis(payload) {
        if (!payload) return;

        // Ensure we always show the loading state rather than an instant cache/heuristic hit to reassure the user
        console.log("[*] Performing Combined Analysis:", payload);
        window.isAnalyzing = true;

        // UI Prep: Disable button and show container
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            if (!isServerOnline) {
                analyzeBtn.innerHTML = '<i class="fas fa-bolt fa-spin"></i> Waking Up Engine...';
            } else {
                analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            }
        }

        resultContainer.style.display = "block"; 
        
        // Hide cards first for loading sequence
        [aiResultCard, fcResultCard, unifiedVerdictCard].forEach(card => card.classList.add("hidden"));
        
        // Reset result elements cleanly
        [aiResult, fcResult, unifiedVerdict].forEach(el => {
            if (!el.innerText || el.innerText === "⏳") {
                el.innerHTML = "⏳";
                el.style.color = "#888";
            }
        });

        try {
            const combinedRes = await fetch(`${API_BASE_URL}/api/analyze-deep`, {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
            }).then(r => r.json());

            if (combinedRes.error) {
                showError(combinedRes.error, payload.type === "text" ? "newsText" : "newsURL");
                resultContainer.style.display = "none";
                return;
            }

            const { ai: aiRes, fc: fcRes, deep: d_res } = combinedRes;

            // Proper Display Sequence
            // First show AI and Fact Check results...
            [aiResultCard, fcResultCard].forEach(card => card.classList.remove("hidden"));
            
            // Allow a small delay before showing the Unified Result to let the user see the analysis completed
            setTimeout(() => {
                displayFinalResults(aiRes, fcRes, d_res);
                console.log("[✅] Analysis complete and synced.");
            }, 800);


        } catch (err) {
            console.error("Analysis Failure:", err);
            showError("Xiriirka backend-ka waa fashilmay (Backend Connection Failed)!", "");
        } finally {
            window.isAnalyzing = false;
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-search-plus"></i> DEEP ANALYSIS';
            }
        }
    }


    function getPayload() {
        const selected = document.querySelector('input[name="inputType"]:checked');
        const inputType = selected.value;
        let data = "";

        if (inputType === "text") {
            if (!newsText) return null;
            data = newsText.value.trim();
            // Skip validation if input contains a link-like structure
            if (!containsLink(data) && (data.split(/\s+/).length < 10 || data.length < 60)) { 
                showError("Fadlan faafaahin badan soo geli si aan kuugu analyse gareeyo", "newsText"); 
                return null; 
            }
        } else {
            if (!newsURL) return null;
            data = newsURL.value.trim();
            if (!data) { showError("Fadlan geli link-ga warka (Please enter the news URL).", "newsURL"); return null; }
        }

        return { type: inputType, data: data };

    }

    // --- [ Event Listeners ] ---
    [analyzeBtn, factCheckBtn].forEach(btn => {
        if (btn) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const payload = getPayload();
                if (payload) performDeepAnalysis(payload);
            });
        }
    });

    if (refreshBtn) {
        refreshBtn.addEventListener('click', (e) => {
            e.preventDefault();
            if (newsText) newsText.value = "";
            if (newsURL) newsURL.value = "";
            resultContainer.style.display = "none";
            [aiResultCard, fcResultCard, unifiedVerdictCard].forEach(card => card.classList.add("hidden"));
            if (unifiedVerdict) unifiedVerdict.innerText = "";
            if (fcReasons) fcReasons.innerHTML = "";
            clearError();

        });
    }

    // ----------------------------
    // 4. Contact Form Handling
    // ----------------------------
    if (submitBtn) {
        submitBtn.addEventListener('click', () => {
            const name = document.getElementById("contactName").value.trim();
            const email = document.getElementById("contactEmail").value.trim();
            const message = document.getElementById("contactMessage").value.trim();

            if (!name || !email || !message) {
                alert("Please fill in all fields.");
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

    // ----------------------------
    // 5. Admin Portal Logic (Safe SPA - No Scroll)
    // ----------------------------
    const adminLoginSection = document.getElementById("adminLoginSection");
    const adminError = document.getElementById("adminError");
    const mainAdminLoginBtn = document.getElementById("mainAdminLoginBtn");
    const startRetrainBtn = document.getElementById('startRetrainBtn');

    function toggleAdminModal(show) {
        if (!adminLoginSection) return;
        if (show) {
            adminLoginSection.classList.remove("hidden");
            adminLoginSection.style.display = "flex";
            if (adminError) adminError.style.display = "none";
            history.replaceState(null, null, '#admin');
        } else {
            adminLoginSection.classList.add("hidden");
            adminLoginSection.style.display = "none";
            if (window.location.hash === '#admin') {
                history.replaceState(null, null, '#home');
                showSection('home');
            }
        }
    }

    // Global toggle for index.html Abort button
    window.closeAdmin = () => toggleAdminModal(false);

    if (mainAdminLoginBtn) {
        mainAdminLoginBtn.addEventListener('click', async () => {
            const user = document.getElementById('adminUsername').value;
            const pass = document.getElementById('adminPassword').value;

            if (!user || !pass) {
                if (adminError) { adminError.innerText = "Please fill in all fields!"; adminError.style.display = 'block'; }
                return;
            }


            try {
                const response = await fetch(`${API_BASE_URL}/api/admin/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: user, password: pass })
                });

                const data = await response.json();
                if (data.success) {
                    localStorage.setItem('adminToken', data.token);
                    window.location.href = 'Admin.html';
                } else {
                    if (adminError) { adminError.innerText = data.message || "Invalid Username or Password!"; adminError.style.display = 'block'; }
                }

            } catch (err) {
                if (adminError) { adminError.innerText = "Error: Ma xirna backend-ka!"; adminError.style.display = 'block'; }
            }
        });
    }

    window.adminLogout = () => {
        localStorage.removeItem('adminToken');
        window.location.href = 'index.html';
    };

    window.switchAdminTab = (tabId) => {
        const tabs = ['overview', 'datasets', 'retrain', 'logs', 'history', 'editor'];
        tabs.forEach(t => {
            const el = document.getElementById(`admin${t.charAt(0).toUpperCase() + t.slice(1)}Tab`);
            if (el) el.classList.add('hidden');
        });
        const target = document.getElementById(`admin${tabId.charAt(0).toUpperCase() + tabId.slice(1)}Tab`);
        if (target) target.classList.remove('hidden');

        // Style active menu
        document.querySelectorAll('.admin-nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.innerText.toLowerCase().includes(tabId)) item.classList.add('active');
        });

        if (tabId === 'datasets') loadAdminDatasets();
        if (tabId === 'logs') loadAdminLogs();
        if (tabId === 'history') loadAdminHistory();
    };

    async function loadAdminStats() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/stats`);
            const data = await res.json();
            const statDs = document.getElementById('statDatasets');
            const statAcc = document.getElementById('statAccuracy');
            if (statDs) statDs.innerText = data.total_datasets;
            if (statAcc) statAcc.innerText = data.model_accuracy;

            const statMessages = document.getElementById('statMessages');
            if (statMessages) statMessages.innerText = data.messages_count;
            const statHistory = document.getElementById('statHistory');
            if (statHistory) statHistory.innerText = data.history_count || 0;
            const statRequests = document.getElementById('statRequests');
            if (statRequests) statRequests.innerText = data.requests_handled;
            loadDashNotifications();
        } catch (err) {
            console.error("Stats Error:", err);
        }
    }

    async function loadAdminDatasets() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/datasets`);
            const data = await res.json();
            const body = document.getElementById('datasetsBody');
            if (body) {
                body.innerHTML = data.map(f => `
                    <tr>
                        <td>${f.name}</td>
                        <td>${f.size} (${f.rows} entries)</td>
                        <td>
                            <button class="admin-btn-login" style="width:auto; padding:5px 15px; background:#10b981;" 
                                onclick="downloadDataset('${f.name}')"><i class="fas fa-download"></i> Download</button>
                        </td>
                    </tr>
                `).join('');
            }
        } catch (err) { console.error("Datasets Error:", err); }
    }

    window.deleteDataset = async (filename) => {
        if (!confirm(`Are you sure you want to delete the dataset '${filename}'? This cannot be undone.`)) return;

        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/dataset/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const data = await res.json();
            if (data.success) {
                alert(data.message);
                loadAdminDatasets();
            } else {
                alert("Khalad: " + data.message);
            }
        } catch (err) {
            alert("Connection Error!");
        }
    };

    window.downloadDataset = (filename) => {
        window.open(`${API_BASE_URL}/api/admin/dataset/download?filename=${filename}`, '_blank');
    };

    window.quickAddEntry = async () => {
        const status = document.getElementById('addEntryStatus');
        const entry = {
            link: document.getElementById('addEntryLink').value,
            title: document.getElementById('addEntryTitle').value,
            Text: document.getElementById('addEntryText').value,
            Subject: document.getElementById('addEntrySubject').value,
            label: document.getElementById('addEntryLabel').value
        };
        const filename = document.getElementById('addEntryFile').value;

        if (!entry.title || !entry.Text) {
            status.innerText = "Please fill in Title and Content!";
            status.style.color = "#ef4444";
            return;
        }


        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/dataset/add_entry`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, entry })
            });
            const data = await res.json();
            status.innerText = data.message;
            status.style.color = "#10b981";
            // Clear fields
            ['addEntryLink', 'addEntryTitle', 'addEntryText', 'addEntrySubject'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = '';
            });
            loadAdminDatasets();
        } catch (err) {
            status.innerText = "Error adding entry!";
            status.style.color = "#ef4444";
        }
    };

    let currentEditorData = { filename: '', columns: [], rows: [] };

    window.openCsvEditor = async (filename) => {
        switchAdminTab('editor');
        const titleEl = document.getElementById('editingFilename');
        if (titleEl) titleEl.innerText = `Loading: ${filename}...`;
        currentEditorData.filename = filename;

        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/dataset/get?filename=${filename}`);
            const data = await safeJson(res);

            if (!res.ok) throw new Error(data.error || `Server error: ${res.status}`);

            if (titleEl) titleEl.innerText = `Editing: ${filename}`;
            currentEditorData.columns = data.columns;
            currentEditorData.rows = data.data;
            renderEditorTable();
        } catch (err) {
            console.error("CSV Load Error:", err);
            if (titleEl) titleEl.innerText = `Error loading ${filename}`;
            alert(`Khalad: ${err.message}`);
        }
    };

    function renderEditorTable() {
        const head = document.getElementById('editorHead');
        const body = document.getElementById('editorBody');
        if (!body) return;

        if (!currentEditorData.columns || !currentEditorData.rows) {
            body.innerHTML = '<tr><td colspan="5">Xogta lama heli karo</td></tr>';
            return;
        }

        if (head) head.innerHTML = `<tr>${currentEditorData.columns.map(c => `<th>${c}</th>`).join('')}<th>Action</th></tr>`;
        body.innerHTML = currentEditorData.rows.length > 0
            ? currentEditorData.rows.map((row, rIdx) => `
                <tr>
                    ${row.map((cell, cIdx) => `
                        <td><input type="text" value="${cell || ''}" onchange="updateEditorCell(${rIdx}, ${cIdx}, this.value)"></td>
                    `).join('')}
                    <td><button class="admin-btn-login" style="width:auto; padding:5px 10px; background:#ef4444;" onclick="deleteEditorRow(${rIdx})"><i class="fas fa-trash"></i></button></td>
                </tr>
            `).reverse().join('')
            : '<tr><td colspan="5" style="padding:20px; text-align:center;">Faylku waa maran yahay. Ku dar saf cusub!</td></tr>';
    }

    window.deleteEditorRow = (idx) => {
        if (!confirm("Are you sure you want to delete this row?")) return;

        currentEditorData.rows.splice(idx, 1);
        renderEditorTable();
    };

    window.updateEditorCell = (rIdx, cIdx, val) => {
        currentEditorData.rows[rIdx][cIdx] = val;
    };

    window.addEditorRow = () => {
        const newRow = new Array(currentEditorData.columns.length).fill('');
        currentEditorData.rows.push(newRow);
        renderEditorTable();
    };

    const saveBtn = document.getElementById('saveDatasetBtn');
    if (saveBtn) {
        saveBtn.onclick = async () => {
            saveBtn.disabled = true;
            saveBtn.innerText = "Saving...";
            try {
                const res = await fetch(`${API_BASE_URL}/api/admin/dataset/save`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filename: currentEditorData.filename,
                        columns: currentEditorData.columns,
                        rows: currentEditorData.rows
                    })
                });
                const data = await res.json();
                alert(data.message);
            } catch (err) {
                alert("Error saving dataset!");
            } finally {
                saveBtn.disabled = false;
                saveBtn.innerText = "Save Changes";
            }
        };
    }

    async function loadAdminLogs() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/logs`);
            const data = await res.json();
            const body = document.getElementById('logsBody');
            if (body) {
                body.innerHTML = data.map(l => `
                    <tr>
                        <td>${l.name}</td>
                        <td>${l.email}</td>
                        <td>${l.message}</td>
                        <td>
                            <button class="admin-btn-login" style="width:auto; padding:5px 15px; margin-right:5px; background:#10b981;" 
                                onclick="openReplyModal('${l.email}', '${l.name.replace(/'/g, "\\'")}')"><i class="fas fa-reply"></i> Reply</button>
                            <button class="admin-btn-login" style="width:auto; padding:5px 15px; background:#ef4444;" 
                                onclick="deleteAdminLog(${l.id})"><i class="fas fa-trash"></i> Delete</button>
                        </td>
                    </tr>
                `).join('');
            }
        } catch (err) { console.error("Logs Error:", err); }
    }

    let historyData = [];

    window.loadAdminHistory = async () => {
        try {
            const body = document.getElementById('historyBody');
            if (!body) return;
            body.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 20px;"><i class="fas fa-spinner fa-spin"></i> Loading...</td></tr>';

            const res = await fetch(`${API_BASE_URL}/api/admin/analysis_history?t=${Date.now()}`);
            historyData = await res.json();

            if (historyData.error) throw new Error(historyData.error);

            if (!Array.isArray(historyData) || historyData.length === 0) {
                body.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 20px;">No history records found.</td></tr>';
                return;
            }

            body.innerHTML = historyData.map((h) => {
                const originalInput = h.original_input || '';
                const nameSnippet = originalInput.length > 50 ? originalInput.substring(0, 50) + "..." : (originalInput || "Unknown");
                const labelUpper = (h.label || "").toUpperCase();
                const isTrusted = labelUpper.includes('TRUSTED') || labelUpper.includes('RASMI') || labelUpper.includes('REAL');

                return `
                <tr>
                    <td title="${originalInput}">${nameSnippet}</td>
                    <td><span style="background: rgba(59,130,246,0.1); color: #3b82f6; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem;">${h.confidence || '-'}</span></td>
                    <td><span style="background: ${isTrusted ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)'}; color: ${isTrusted ? '#10b981' : '#ef4444'}; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem;">${h.label || '-'}</span></td>
                    <td style="color: #9ca3af; font-size: 0.85rem;">${h.date || '-'}</td>
                    <td>
                        <button onclick="viewHistoryItem(${h.id})" class="admin-btn-login" style="width:auto; padding:5px 10px; margin-right:5px; background:#3b82f6;" title="View Detail"><i class="fas fa-eye"></i></button>
                        <button onclick="deleteHistoryItem(${h.id})" class="admin-btn-login" style="width:auto; padding:5px 10px; background:#ef4444;" title="Delete"><i class="fas fa-trash"></i></button>
                    </td>
                </tr>
            `}).join('');
        } catch (err) {
            console.error("History Error:", err);
            const b = document.getElementById('historyBody');
            if (b) b.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #ef4444; padding: 20px;">Error loading history</td></tr>`;
        }
    };

    window.viewHistoryItem = (id) => {
        const item = historyData.find(i => i.id === id);
        if (!item) return;

        document.getElementById("modalOriginalInput").innerText = item.original_input;
        document.getElementById("modalExtractedText").innerText = item.extracted_text;
        document.getElementById("modalResult").innerText = `${item.label} (${item.confidence})`;
        document.getElementById("modalResult").style.color = (item.label.includes('Trusted') || item.label.includes('Rasmi')) ? "#10b981" : "#ef4444";
        document.getElementById("modalDate").innerText = item.date;

        const modal = document.getElementById("historyDetailModal");
        if (modal) {
            modal.style.display = "flex";
            modal.classList.remove("hidden");
        }
    };

    window.closeHistoryModal = () => {
        const modal = document.getElementById("historyDetailModal");
        if (modal) {
            modal.style.display = "none";
            modal.classList.add("hidden");
        }
    };

    window.deleteHistoryItem = async (id) => {
        if (!confirm("Are you sure you want to delete this analysis?")) return;

        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/analysis_history/delete`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id: id })
            });
            const data = await res.json();
            if (data.success) loadAdminHistory();
            else alert("Khalad: " + data.message);
        } catch (err) { alert("Connection Error!"); }
    };

    window.clearAllHistory = async () => {
        if (!confirm("Are you sure you want to delete ALL analysis history?")) return;

        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/analysis_history/clear`, {
                method: "POST",
                headers: { "Content-Type": "application/json" }
            });
            const data = await res.json();
            if (data.success) loadAdminHistory();
            else alert("Khalad: " + data.message);
        } catch (err) { alert("Connection Error!"); }
    };

    async function loadDashNotifications() {
        const listDiv = document.getElementById('dashNotificationList');
        if (!listDiv) return;
        try {
            const res = await fetch(`${API_BASE_URL}/api/admin/logs`);
            const data = await res.json();
            if (data.length === 0) {
                listDiv.innerHTML = '<div style="padding: 20px; color: #9ca3af; font-size: 0.9rem; text-align: center;">No notifications</div>';
                return;
            }
            const latest = data.slice(0, 4);
            const badge = document.getElementById('navNotifBadge');
            if (badge) {
                badge.style.display = data.length > 0 ? 'flex' : 'none';
                badge.innerText = data.length > 9 ? '9+' : data.length;
            }
            listDiv.innerHTML = latest.map(l => `
                <div style="padding: 15px 20px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: start; gap: 12px;">
                    <i class="fas fa-inbox" style="color: #3b82f6; margin-top: 4px;"></i>
                    <div style="flex: 1;">
                        <h4 style="margin: 0; font-size: 0.9rem; color: #fff;">New message from ${l.name}</h4>
                        <p style="margin:5px 0 0 0; color:#9ca3af; font-size:0.8rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${l.message}</p>
                    </div>
                </div>
            `).join('');
        } catch (err) { console.error("Notif Error:", err); }
    }

    window.showToast = (message, type = "success") => {
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.style.cssText = 'position:fixed; bottom:20px; right:20px; z-index:99999; display:flex; flex-direction:column; gap:10px;';
            document.body.appendChild(toastContainer);
        }
        const toast = document.createElement('div');
        const bg = type === 'success' ? '#10b981' : '#ef4444';
        toast.style.cssText = `background:${bg}; color:#fff; padding:12px 20px; border-radius:8px; box-shadow:0 10px 15px rgba(0,0,0,0.3); font-size:14px; opacity:0; transform:translateY(20px); transition:all 0.3s;`;
        toast.innerText = message;
        toastContainer.appendChild(toast);
        setTimeout(() => { toast.style.opacity = '1'; toast.style.transform = 'translateY(0)'; }, 10);
        setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
    };

    if (startRetrainBtn) {
        startRetrainBtn.addEventListener('click', async () => {
            const status = document.getElementById('retrainStatus');
            if (!status) return;
            startRetrainBtn.disabled = true;
            status.innerHTML = '<i class="fas fa-sync fa-spin"></i> Tababarku waa billowday...';
            try {
                const res = await fetch(`${API_BASE_URL}/api/admin/retrain`, { method: 'POST' });
                const data = await res.json();
                if (!data.success) { status.innerText = "Error: " + data.message; startRetrainBtn.disabled = false; return; }

                const poll = setInterval(async () => {
                    const sRes = await fetch(`${API_BASE_URL}/api/admin/retrain_status`);
                    const sData = await sRes.json();
                    if (!sData.is_training) {
                        clearInterval(poll);
                        status.innerHTML = '<i class="fas fa-check-circle"></i> Tababarkii waa dhamaaday!';
                        startRetrainBtn.disabled = false;
                        loadAdminStats();
                    }
                }, 3000);
            } catch (err) { status.innerText = "Server error!"; startRetrainBtn.disabled = false; }
        });
    }

    // --- [ Final Initialization ] ---
    function handleRouting() {
        const hash = window.location.hash.substring(1);
        if (hash === 'admin') {
            toggleAdminModal(true);
        } else if (hash && document.getElementById(hash)) {
            showSection(hash);
        } else {
            // Default to home but don't force it if we are in admin mode
            if (!adminLoginSection || adminLoginSection.classList.contains('hidden')) {
                showSection('home');
            }
        }
    }

    window.addEventListener('hashchange', handleRouting);
    handleRouting();
});
