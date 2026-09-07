// Smart Agriculture Advisor JavaScript

// Welcome message
console.log("Smart Agriculture Advisor Loaded Successfully");

// Alert animation
document.addEventListener("DOMContentLoaded", function () {

    const alerts = document.querySelectorAll(".alert");

    alerts.forEach(function(alert) {

        setTimeout(() => {
            alert.style.display = "none";
        }, 3000);

    });

});

// Form Validation
function validateForm() {

    let inputs = document.querySelectorAll("input[required]");

    for (let input of inputs) {

        if (input.value.trim() === "") {

            alert("Please fill all required fields");
            return false;
        }
    }

    return true;
}

// Smooth Scroll Effect
document.querySelectorAll('a[href^="#"]').forEach(anchor => {

    anchor.addEventListener('click', function (e) {

        e.preventDefault();

        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });

    });

});

// Dashboard Card Hover Animation
const cards = document.querySelectorAll(".card");

cards.forEach(card => {

    card.addEventListener("mouseover", () => {

        card.style.transform = "scale(1.03)";
        card.style.transition = "0.3s";

    });

    card.addEventListener("mouseout", () => {

        card.style.transform = "scale(1)";

    });

});

// Current Date and Time
function showDateTime() {

    let date = new Date();

    let dateTime = date.toLocaleString();

    let element = document.getElementById("datetime");

    if (element) {
        element.innerHTML = dateTime;
    }

}

setInterval(showDateTime, 1000);

// =========================
// Light / Dark Theme Toggle
// =========================
// Theme is applied instantly on <html data-theme="..."> by an inline
// script in base.html's <head> (so there's no flash-of-wrong-theme on
// load). This block just wires up the toggle button + keeps the
// button's icon/label and localStorage in sync.
(function () {

    const btn = document.getElementById("theme-toggle-btn");
    const icon = document.getElementById("theme-toggle-icon");
    const label = document.getElementById("theme-toggle-label");
    const i18n = window.THEME_I18N || { light: "Light", dark: "Dark" };

    function currentTheme() {
        return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    }

    function updateButton(theme) {
        if (!icon || !label) return;
        if (theme === "dark") {
            icon.textContent = "☀️";
            label.textContent = i18n.light; // button shows the theme you'd switch TO
        } else {
            icon.textContent = "🌙";
            label.textContent = i18n.dark;
        }
    }

    updateButton(currentTheme());

    if (btn) {
        btn.addEventListener("click", function () {
            const next = currentTheme() === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", next);
            try {
                localStorage.setItem("theme", next);
            } catch (e) {
                // localStorage unavailable (private mode, etc.) — theme
                // still applies for this page view, just won't persist.
            }
            updateButton(next);
        });
    }

})();