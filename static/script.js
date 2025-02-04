document.addEventListener("DOMContentLoaded", function () {
    const toggleThemeBtn = document.getElementById("toggleTheme");
    const body = document.body;
    const urlInput = document.getElementById("urlInput");
    const copyUrlBtn = document.getElementById("copyUrl");
    const resultSection = document.getElementById("result");
    const loadingSection = document.getElementById("loading");
    const historyList = document.getElementById("historyList");

    // Load Dark Mode Preference
    if (localStorage.getItem("darkMode") === "enabled") {
        body.classList.add("dark-mode");
    }

    // Toggle Dark Mode
    toggleThemeBtn.addEventListener("click", function () {
        body.classList.toggle("dark-mode");
        localStorage.setItem("darkMode", body.classList.contains("dark-mode") ? "enabled" : "disabled");
    });

    // Copy URL to Clipboard
    copyUrlBtn.addEventListener("click", function () {
        if (urlInput.value.trim() !== "") {
            navigator.clipboard.writeText(urlInput.value);
            alert("URL copied to clipboard!");
        } else {
            alert("Please enter a URL first!");
        }
    });

    // Check Website Function (connect to API)
    window.checkWebsite = async function () {
        const url = urlInput.value.trim();
        if (url === "") {
            resultSection.innerHTML = "<p style='color: red;'>Please enter a valid URL!</p>";
            return;
        }

        // Show Loading Animation
        loadingSection.style.display = "block";
        resultSection.innerHTML = "<p style='color: yellow;'>🔄 Checking...</p>";

        // Start the timer
        console.time("API Response Time");

        try {
            const response = await fetch("http://127.0.0.1:8001/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();
            
            if (data.prediction === "Phishing") {
                resultSection.innerHTML = `<p style="color: red;">⚠️ Warning! This might be a phishing website.</p>`;
            } else {
                resultSection.innerHTML = `<p style="color: green;">✅ This website appears to be safe.</p>`;
            }

            // Hide loading animation after receiving the result
            loadingSection.style.display = "none";

            // Save URL to history
            saveToHistory(url, data.prediction === "Phishing");

            // End the timer and display the elapsed time
            console.timeEnd("API Response Time");
        } catch (error) {
            resultSection.innerHTML = `<p style='color: orange;'>❌ Error checking website. Try again later.</p>`;
            console.error("Error:", error);
            console.timeEnd("API Response Time");  // End timer even if there's an error
        }
    };

    // Save Checked URLs to History
    function saveToHistory(url, isPhishing) {
        let history = JSON.parse(localStorage.getItem("urlHistory")) || [];
        if (!history.some(entry => entry.url === url)) {
            history.push({ url, status: isPhishing ? "⚠️ Phishing" : "✅ Safe" });
            localStorage.setItem("urlHistory", JSON.stringify(history));
        }
        updateHistoryDisplay();
    }

    // Display URL History
    function updateHistoryDisplay() {
        historyList.innerHTML = "";
        let history = JSON.parse(localStorage.getItem("urlHistory")) || [];

        history.slice().reverse().forEach(entry => {
            const li = document.createElement("li");
            li.innerHTML = `<span>${entry.url}</span> <span>${entry.status}</span>`;
            historyList.appendChild(li);
        });
    }

    // Load History on Page Load
    updateHistoryDisplay();
});

// Fix: Correct button ID for the event listener
document.getElementById("checkWebsiteBtn").addEventListener("click", async () => {
    window.checkWebsite();
});
