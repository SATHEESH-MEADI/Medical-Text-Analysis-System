import os
import streamlit.components.v1 as components

# Declare your custom component with the local path
_COMPONENT_DIR = os.path.join(os.path.dirname('/Users/satheesh/Desktop/NLP/theme_toggle_component/frontend/'), "frontend")


# Declare the component
toggle_theme_component = components.declare_component(
    "toggle_theme_component",
    path=_COMPONENT_DIR
)

# Wrapper function to call the component in the Streamlit app
def theme_toggle():
    return toggle_theme_component()



