// =====================================================
// AI Chatbot Widget (Flask backend + Gemini API)
// =====================================================

document.addEventListener("DOMContentLoaded", function () {

    const toggleBtn = document.getElementById("chatbot-toggle-btn");
    const closeBtn = document.getElementById("chatbot-close-btn");
    const panel = document.getElementById("chatbot-panel");
    const form = document.getElementById("chatbot-form");
    const input = document.getElementById("chatbot-input");
    const messagesBox = document.getElementById("chatbot-messages");
    const typingIndicator = document.getElementById("chatbot-typing");

    const i18n = window.CHATBOT_I18N || { greeting: "Hello!", error: "Something went wrong.", lang: "en" };

    let greeted = false;

    function addMessage(text, sender) {
        const bubble = document.createElement("div");
        bubble.className = "chatbot-msg " + sender;
        bubble.textContent = text;
        messagesBox.appendChild(bubble);
        messagesBox.scrollTop = messagesBox.scrollHeight;
        return bubble;
    }

    function openPanel() {
        panel.classList.remove("d-none");
        if (!greeted) {
            addMessage(i18n.greeting, "bot");
            greeted = true;
        }
        input.focus();
    }

    toggleBtn.addEventListener("click", function () {
        if (panel.classList.contains("d-none")) {
            openPanel();
        } else {
            panel.classList.add("d-none");
        }
    });

    closeBtn.addEventListener("click", function () {
        panel.classList.add("d-none");
    });

    form.addEventListener("submit", function (e) {
        e.preventDefault();

        const message = input.value.trim();
        if (!message) return;

        addMessage(message, "user");
        input.value = "";
        input.focus();

        typingIndicator.classList.remove("d-none");

        fetch("/api/chatbot", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                lang: i18n.lang
            })
        })
            .then(function (res) {
                return res.json();
            })
            .then(function (data) {
                typingIndicator.classList.add("d-none");
                addMessage(data.reply || i18n.error, "bot");
            })
            .catch(function () {
                typingIndicator.classList.add("d-none");
                addMessage(i18n.error, "bot");
            });
    });

});
