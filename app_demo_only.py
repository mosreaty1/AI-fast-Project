"""
Streamlit Demo App - No PyTorch Required
For Windows users who can't install PyTorch
"""
import streamlit as st
import pandas as pd

# Mock predictions for demo purposes
MOCK_PREDICTIONS = {
    ("I really love this product!", "positive"): "really love",
    ("This is terrible and I hate it", "negative"): "terrible",
    ("I went to the store today", "neutral"): "went to the store today",
    ("Best purchase ever!", "positive"): "Best",
    ("Worst service I've ever had", "negative"): "Worst service",
}

def mock_predict(text: str, sentiment: str) -> str:
    """Mock prediction function."""
    # Check if we have this exact example
    if (text, sentiment) in MOCK_PREDICTIONS:
        return MOCK_PREDICTIONS[(text, sentiment)]

    # Otherwise, return a simple heuristic
    words = text.split()
    sentiment_words = {
        "positive": ["love", "great", "best", "amazing", "wonderful", "excellent", "good", "like"],
        "negative": ["hate", "terrible", "worst", "awful", "bad", "disappointing", "poor"],
        "neutral": words  # Return all for neutral
    }

    # Find sentiment words in text
    for word in words:
        if word.lower() in sentiment_words.get(sentiment, []):
            return word

    # Default: return first few words
    return " ".join(words[:3])


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Tweet Sentiment Extraction - DEMO",
        page_icon="🐦",
        layout="wide"
    )

    # Warning banner
    st.warning("⚠️ **DEMO MODE**: This is a demonstration version without the actual AI model. Mock predictions are shown for UI demonstration purposes only.")

    # Title and description
    st.title("🐦 Tweet Sentiment Extraction - DEMO")
    st.markdown("""
    **AIE417 Selected Topics in AI - Fall 2025**

    This is a **demo version** for Windows users who encounter PyTorch installation issues.
    For the full version with the trained model, please resolve the PyTorch installation or use Linux/Mac.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Demo Configuration")
    st.sidebar.info("This demo uses mock predictions. The actual model would use these parameters:")

    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    num_beams = st.sidebar.slider("Number of Beams", 1, 10, 4, 1)

    # Main interface
    st.header("📝 Extract Sentiment Phrases")

    col1, col2 = st.columns(2)

    with col1:
        text_input = st.text_area(
            "Tweet Text",
            height=150,
            placeholder="Enter a tweet here...",
            value="I really love this product!"
        )

    with col2:
        sentiment = st.selectbox(
            "Sentiment",
            options=["positive", "negative", "neutral"],
            index=0
        )

    # Example tweets
    st.subheader("📋 Example Tweets")
    examples = {
        "Positive": {
            "text": "I really love this product!",
            "sentiment": "positive"
        },
        "Negative": {
            "text": "This is terrible and I hate it",
            "sentiment": "negative"
        },
        "Neutral": {
            "text": "I went to the store today",
            "sentiment": "neutral"
        }
    }

    example_cols = st.columns(3)
    for i, (label, example) in enumerate(examples.items()):
        with example_cols[i]:
            if st.button(f"Use {label} Example"):
                st.session_state.example_text = example["text"]
                st.session_state.example_sentiment = example["sentiment"]
                st.rerun()

    # Use example if selected
    if 'example_text' in st.session_state:
        text_input = st.session_state.example_text
        sentiment = st.session_state.example_sentiment

    # Generate prediction
    if st.button("🎯 Extract Sentiment Phrase (DEMO)", type="primary"):
        if not text_input:
            st.warning("Please enter some text!")
        else:
            with st.spinner("Generating mock prediction..."):
                prediction = mock_predict(text_input, sentiment)

                # Display results
                st.success("✅ Extraction Complete! (DEMO)")

                st.subheader("📊 Results")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.markdown("**Original Tweet:**")
                    st.info(text_input)

                    st.markdown("**Sentiment:**")
                    sentiment_color = {
                        "positive": "🟢",
                        "negative": "🔴",
                        "neutral": "⚪"
                    }
                    st.info(f"{sentiment_color[sentiment]} {sentiment.upper()}")

                with result_col2:
                    st.markdown("**Extracted Phrase (MOCK):**")
                    st.success(prediction)

                    st.markdown("**Highlighted in Text:**")
                    if prediction.lower() in text_input.lower():
                        highlighted = text_input.replace(
                            prediction,
                            f"**:green[{prediction}]**"
                        )
                        st.markdown(highlighted)
                    else:
                        st.markdown(f"{text_input}\n\n*(Extracted: **:green[{prediction}]**)*")

                st.info("💡 **Note**: This is a mock prediction. The actual trained model would provide more accurate results.")

    # Batch processing demo
    st.header("📦 Batch Processing (DEMO)")
    st.info("In the full version, you can upload CSV files for batch processing.")

    # Model information
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        ### Why Demo Mode?

        You're seeing this demo because PyTorch couldn't be loaded on your system. Common causes:

        1. **Python version too new** (3.14) - PyTorch supports up to 3.12
        2. **Missing Visual C++ Redistributables** on Windows
        3. **Incompatible PyTorch installation**

        ### To Use the Full Version:

        1. **Downgrade Python** to 3.11 or 3.12
        2. **Install Visual C++ Redistributables**
        3. **Reinstall PyTorch** properly for Windows

        See the installation guide for details.

        ### What's Different in Full Version:

        - ✅ Actual FLAN-T5 model with PEFT/LoRA
        - ✅ Real predictions (not mocks)
        - ✅ Configurable generation parameters
        - ✅ Batch CSV processing
        - ✅ 0.72 Jaccard score accuracy

        ### Current Demo Features:

        - ✅ UI demonstration
        - ✅ Example predictions
        - ✅ Interface testing
        - ❌ No actual AI model
        - ❌ Mock predictions only
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built for AIE417 - Selected Topics in AI</p>
        <p>Dr. Laila Shoukry | Fall 2025</p>
        <p><b>DEMO MODE - PyTorch Not Available</b></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
