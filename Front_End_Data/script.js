document.addEventListener('DOMContentLoaded', () => {

    // ================= API URL =================
    const API_URL = "https://tafaftire-detection-system.onrender.com";

    // ================= ELEMENTS =================
    const predictBtn = document.getElementById("predictBtn");
    const submitBtn = document.querySelector(".submit-btn");
    const refreshBtn = document.getElementById("refreshBtn");

    const resultDiv = document.getElementById("result");
    const confidenceDiv = document.getElementById("confidence");

    const newsText = document.getElementById("newsText");
    const newsURL = document.getElementById("newsURL");

    const textInput = document.getElementById("textInput");
    const urlInput = document.getElementById("urlInput");

    const navLinks = document.querySelectorAll(".nav-links a");
    const sections = document.querySelectorAll("section");
    const hamburger = document.querySelector(".hamburger");
    const navMenu = document.querySelector(".nav-links");

    // ================= HEALTH CHECK =================
    fetch(`${API_URL}/`)
        .then(res => {
            if (!res.ok) throw new Error("Server error");
            return res.json();
        })
        .then(data => console.log("✅ Server:", data))
        .catch(err => console.warn("⚠️ Server ma shaqeynayo.", err));

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
            if (link.getAttribute("href") === "#" + id) link.classList.add("active");
        });

        if (navMenu) navMenu.classList.remove("active");
    }

    navLinks.forEach(link => {
        link.addEventListener("click", e => {
            e.preventDefault();
            const id = link.getAttribute("href").substring(1);
            showSection(id);
        });
    });

    if (hamburger && navMenu) {
        hamburger.addEventListener("click", () => navMenu.classList.toggle("active"));
    }

    showSection("home");

    // ================= INPUT TOGGLE =================
    document.querySelectorAll('input[name="inputType"]').forEach(radio => {
        radio.addEventListener("change", () => {
            if (!textInput || !urlInput) return;
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
        predictBtn.addEventListener("click", async () => {

            const selected = document.querySelector('input[name="inputType"]:checked');
            if (!selected) return;

            let data = selected.value === "text" ? (newsText?.value.trim() || "") : (newsURL?.value.trim() || "");

            if (selected.value === "text") {
                if (data.length < 20) {
                    resultDiv.innerText = "❌ Qoraal aad u gaaban.";
                    return;
                }
                if (containsLink(data)) {
                    resultDiv.innerText = "❌ Text mode laguma ogola link.";
                    return;
                }
            } else {
                if (!isURL(data)) {
                    resultDiv.innerText = "❌ URL sax ah geli.";
                    return;
                }
            }

            resultDiv.innerText = "⏳ Analyzing...";
            confidenceDiv.innerText = "";

            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: data })
                });

                if (!response.ok) throw new Error("Server response error");

                const res = await response.json();

                if (res.error) resultDiv.innerText = "❌ " + res.error;
                else {
                    const isReal = res.prediction.toUpperCase().includes("REAL");
                    resultDiv.innerText = isReal ? "WAR RUN AH" : "WAR BEEN AH";
                    resultDiv.style.color = isReal ? "#2ecc71" : "#e74c3c";
                    confidenceDiv.innerText = "Kalsoonida: " + (res.confidence || "N/A");
                }

            } catch (err) {
                resultDiv.innerText = "❌ Connection Error";
                console.error(err);
            }
        });
    }

    // ================= RESET =================
    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => {
            if (newsText) newsText.value = "";
            if (newsURL) newsURL.value = "";
            resultDiv.innerText = "";
            confidenceDiv.innerText = "";
        });
    }

    // ================= CONTACT =================
    if (submitBtn) {
        submitBtn.addEventListener("click", async () => {
            const nameEl = document.getElementById("contactName");
            const emailEl = document.getElementById("contactEmail");
            const messageEl = document.getElementById("contactMessage");

            const name = nameEl?.value.trim() || "";
            const email = emailEl?.value.trim() || "";
            const message = messageEl?.value.trim() || "";

            if (!name || !email || !message) {
                alert("Fadlan buuxi meelaha banaan.");
                return;
            }

            try {
                const response = await fetch(`${API_URL}/contact`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name, email, message })
                });

                if (!response.ok) throw new Error("Server error");

                const res = await response.json();

                if (res.status === "Success") {
                    alert(res.message);
                    nameEl.value = "";
                    emailEl.value = "";
                    messageEl.value = "";
                } else {
                    alert("Error: " + (res.error || "Unknown error"));
                }

            } catch (error) {
                alert("❌ Connection Error");
                console.error(error);
            }
        });
    }

});
