document.addEventListener('DOMContentLoaded', () => {

    const API_BASE_URL = 'https://tafaftire-detection-system.onrender.com';

    const predictBtn = document.getElementById("predictBtn");
    const submitBtn = document.querySelector('.submit-btn');
    const refreshBtn = document.getElementById("refreshBtn");

    const resultDiv = document.getElementById("result");
    const confidenceDiv = document.getElementById("confidence");
    const newsText = document.getElementById("newsText");
    const newsURL = document.getElementById("newsURL");

    const textInput = document.getElementById("textInput");
    const urlInput = document.getElementById("urlInput");

    const navLinks = document.querySelectorAll('.nav-links a');
    const sections = document.querySelectorAll('section');
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-links');

    // ================= HEALTH CHECK =================
    fetch(`${API_BASE_URL}/`)
        .then(res => res.json())
        .then(data => {
            if (data.status === "OK") {
                console.log("✅ Server Online");
            }
        })
        .catch(() => {
            console.warn("⚠️ Server ma shaqeynayo.");
        });

    // ================= NAVIGATION =================
    function showSection(id) {
        sections.forEach(sec => sec.style.display = "none");
        const target = document.getElementById(id);
        if (target) {
            target.style.display = "block";
            window.scrollTo({ top: 0, behavior: "smooth" });
        }
        navLinks.forEach(link => {
            link.classList.remove("active");
            if (link.getAttribute("href") === "#" + id) {
                link.classList.add("active");
            }
        });
        if (navMenu) navMenu.classList.remove("active");
    }

    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const id = link.getAttribute("href").substring(1);
            showSection(id);
        });
    });

    if (hamburger) {
        hamburger.addEventListener("click", () => {
            navMenu.classList.toggle("active");
        });
    }

    showSection("home");

    // ================= INPUT TOGGLE =================
    document.querySelectorAll('input[name="inputType"]').forEach(radio => {
        radio.addEventListener("change", () => {
            if (radio.value === "text") {
                textInput.classList.remove("hidden");
                urlInput.classList.add("hidden");
            } else {
                textInput.classList.add("hidden");
                urlInput.classList.remove("hidden");
            }
        });
    });

    // ================= VALIDATION =================
    function isURL(text) {
        return /^(https?:\/\/)/i.test(text.trim());
    }

    function containsLink(text) {
        return /(https?:\/\/[^\s]+|www\.[^\s]+)/i.test(text);
    }

    // ================= PREDICT =================
    if (predictBtn) {
        predictBtn.addEventListener("click", () => {

            const selected = document.querySelector('input[name="inputType"]:checked');
            const inputType = selected.value;

            let data = "";

            if (inputType === "text") {
                data = newsText.value.trim();

                if (data.length < 20) {
                    resultDiv.innerText = "❌ Qoraal aad u gaaban.";
                    return;
                }

                if (containsLink(data)) {
                    resultDiv.innerText = "❌ Text mode laguma ogola link.";
                    return;
                }

            } else {
                data = newsURL.value.trim();

                if (!isURL(data)) {
                    resultDiv.innerText = "❌ URL sax ah geli.";
                    return;
                }
            }

            resultDiv.innerText = "⏳ Analyzing...";
            confidenceDiv.innerText = "";

            // ✅ JSON FIXED HERE
            fetch(`${API_BASE_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: data })
            })
            .then(res => res.json())
            .then(res => {
                if (res.error) {
                    resultDiv.innerText = "❌ " + res.error;
                } else {
                    const isReal = res.prediction.includes("REAL");
                    resultDiv.innerText = isReal ? "WAR RUN AH" : "WAR BEEN AH";
                    resultDiv.style.color = isReal ? "#2ecc71" : "#e74c3c";
                    confidenceDiv.innerText = "Kalsoonida: " + res.confidence;
                }
            })
            .catch(() => {
                resultDiv.innerText = "❌ Connection Error";
            });
        });
    }

    // ================= RESET BUTTON =================
    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => {
            newsText.value = "";
            newsURL.value = "";
            resultDiv.innerText = "";
            confidenceDiv.innerText = "";
        });
    }

    // ================= CONTACT =================
    if (submitBtn) {
        submitBtn.addEventListener("click", () => {

            const name = document.getElementById("contactName").value.trim();
            const email = document.getElementById("contactEmail").value.trim();
            const message = document.getElementById("contactMessage").value.trim();

            if (!name || !email || !message) {
                alert("Fadlan buuxi meelaha banaan.");
                return;
            }

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
                    alert("Error: " + res.error);
                }
            })
            .catch(() => {
                alert("Connection Error");
            });
        });
    }

});

