function toggleTheme() {
    const currentTheme = document.body.getAttribute("data-theme") || "light";
    const newTheme = currentTheme === "light" ? "dark" : "light";

    // Apply the new theme
    document.body.setAttribute("data-theme", newTheme);

    // Send the new theme value back to Streamlit
    Streamlit.setComponentValue(newTheme);
}

// Set up event listener for Streamlit re-renders
Streamlit.onRender(() => {
    const theme = Streamlit.getOptions()?.theme || "light";
    document.body.setAttribute("data-theme", theme);
});
