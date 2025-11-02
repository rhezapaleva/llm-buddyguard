# app.py
import streamlit as st
import time
from src.models.baseline import BaselineModel
from src.models.ollama_baseline import OllamaBaselineModel
from src.models.frontier import FrontierModel
from src.guardrails import EducationalGuardrails
from src.evaluation import ModelEvaluator
import os

# Page config
st.set_page_config(
    page_title="LLM BuddyGuard - O-Level Tutor",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_models():
    """Load models with caching"""
    # Try Ollama baseline first (faster and truly local)
    try:
        baseline = OllamaBaselineModel()  # Try llama3:latest by default
        baseline_loaded = True
        baseline_type = "ollama"
    except Exception as e:
        st.warning(f"Ollama baseline model not loaded: {e}")
        # Fallback to HuggingFace baseline
        try:
            baseline = BaselineModel()
            baseline_loaded = True
            baseline_type = "huggingface"
        except Exception as e2:
            st.warning(f"HuggingFace baseline model also failed: {e2}")
            baseline = None
            baseline_loaded = False
            baseline_type = None
    
    try:
        frontier = FrontierModel()
        frontier_loaded = True
    except Exception as e:
        st.warning(f"Frontier model not loaded: {e}")
        frontier = None
        frontier_loaded = False
    
    return baseline, frontier, baseline_loaded, frontier_loaded, baseline_type

baseline_model, frontier_model, baseline_ok, frontier_ok, baseline_type = load_models()
guardrails = EducationalGuardrails()
evaluator = ModelEvaluator()

# Sidebar
st.sidebar.title("Settings")

# Show baseline model info
if baseline_ok:
    model_info = baseline_model.get_model_info() if hasattr(baseline_model, 'get_model_info') else {}
    baseline_label = f"Baseline ({baseline_type}: {model_info.get('model_name', 'Unknown')})"
else:
    baseline_label = "Baseline (Not Available)"

model_choice = st.sidebar.radio(
    "Choose Model:",
    [baseline_label, "Frontier (GPT-4o)", "Compare Both"],
    disabled=not (baseline_ok or frontier_ok)
)

subject = st.sidebar.selectbox(
    "Subject:",
    ["Mathematics", "Science", "English"]
)

show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=False)

# Performance settings
st.sidebar.divider()
st.sidebar.subheader("⚡ Performance")
if baseline_type == "ollama":
    st.sidebar.success("🚀 **Fast**: Using local Ollama model - responses in seconds!")
    if baseline_ok:
        model_info = baseline_model.get_model_info()
        st.sidebar.info(f"Model: {model_info.get('model_name', 'Unknown')}")
elif baseline_type == "huggingface":
    st.sidebar.info("💡 **Tip**: Baseline model runs on CPU and may take 15-30 seconds. For faster responses, use the Frontier model (requires OpenAI API key).")
else:
    st.sidebar.warning("⚠️ No baseline model available")

# Reference answer for accuracy calculation
st.sidebar.divider()
st.sidebar.subheader("📊 Accuracy Evaluation")
reference_answer = st.sidebar.text_area(
    "Reference Answer (optional)",
    placeholder="Enter the expected/correct answer to calculate accuracy...",
    help="If provided, cosine similarity will be calculated between the model's response and this reference answer."
)

# Main interface
st.title("LLM BuddyGuard - O-Level Tutor")
st.markdown("Your AI study companion for Singapore O-Level examinations")

# Model status
col1, col2 = st.columns(2)
with col1:
    status = "Ready" if baseline_ok else "Not Loaded"
    st.metric("Baseline Model", status)
with col2:
    status = "Ready" if frontier_ok else "Not Loaded"
    st.metric("Frontier Model", status)

st.divider()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show accuracy score if available
        if "accuracy_cosine" in message:
            accuracy_score = message["accuracy_cosine"]
            st.metric(
                "Accuracy (Cosine)", 
                f"{accuracy_score:.3f}",
                help="Cosine similarity between model response and reference answer (0.0 - 1.0)"
            )
        
        if "metrics" in message and show_metrics:
            with st.expander("Evaluation Metrics"):
                st.json(message["metrics"])

# Chat input
if prompt := st.chat_input("Ask your O-Level question..."):
    # Apply guardrails
    guardrail_result = guardrails.apply_guardrails(prompt)
    
    if not guardrail_result["allowed"]:
        st.error(guardrail_result["message"])
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show warning if answer-seeking detected
        if guardrail_result["message"] != "Prompt approved":
            st.warning(guardrail_result["message"])
        
        # Generate response
        with st.chat_message("assistant"):
            if model_choice == "Compare Both":
                if baseline_ok and frontier_ok:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Baseline Model**")
                        with st.spinner("Generating..."):
                            baseline_result = baseline_model.generate(
                                prompt, 
                                context=subject, 
                                reference_answer=reference_answer.strip() if reference_answer.strip() else None
                            )
                            st.markdown(baseline_result["response"])
                            
                            # Show accuracy if available
                            if "accuracy_cosine" in baseline_result:
                                st.metric(
                                    "Accuracy (Cosine)", 
                                    f"{baseline_result['accuracy_cosine']:.3f}",
                                    help="Cosine similarity with reference answer"
                                )
                            
                            if show_metrics:
                                metrics = evaluator.evaluate_response(
                                    baseline_result["response"],
                                    reference_answer=reference_answer.strip() if reference_answer.strip() else None
                                )
                                with st.expander("Metrics"):
                                    st.json(metrics)
                    
                    with col2:
                        st.markdown("**Frontier Model**")
                        with st.spinner("Generating..."):
                            frontier_result = frontier_model.generate(
                                prompt, 
                                subject=subject,
                                reference_answer=reference_answer.strip() if reference_answer.strip() else None
                            )
                            st.markdown(frontier_result["response"])
                            
                            # Show accuracy if available
                            if "accuracy_cosine" in frontier_result:
                                st.metric(
                                    "Accuracy (Cosine)", 
                                    f"{frontier_result['accuracy_cosine']:.3f}",
                                    help="Cosine similarity with reference answer"
                                )
                            
                            if show_metrics:
                                metrics = evaluator.evaluate_response(
                                    frontier_result["response"],
                                    reference_answer=reference_answer.strip() if reference_answer.strip() else None
                                )
                                with st.expander("Metrics"):
                                    st.json(metrics)
                else:
                    st.error("Both models must be loaded for comparison mode")
                    
            elif model_choice.startswith("Baseline") and baseline_ok:
                spinner_text = "🚀 Generating with local Ollama model..." if baseline_type == "ollama" else "🤖 Baseline model thinking... (This may take 15-30 seconds on CPU)"
                
                with st.spinner(spinner_text):
                    start_time = time.time()
                    result = baseline_model.generate(
                        prompt, 
                        context=subject, 
                        reference_answer=reference_answer.strip() if reference_answer.strip() else None
                    )
                    generation_time = time.time() - start_time
                    
                    st.markdown(result["response"])
                    
                    # Show generation time
                    st.caption(f"⏱️ Generated in {generation_time:.1f} seconds")
                    
                    # Show accuracy if available
                    if "accuracy_cosine" in result:
                        st.metric(
                            "Accuracy (Cosine)", 
                            f"{result['accuracy_cosine']:.3f}",
                            help="Cosine similarity with reference answer"
                        )
                    
                    # Show model info
                    if hasattr(baseline_model, 'get_model_info'):
                        model_info = baseline_model.get_model_info()
                        st.caption(f"🤖 Model: {model_info.get('model_name', 'Unknown')}")
                    
                    # Prepare message with accuracy for session state
                    message_content = {
                        "role": "assistant",
                        "content": result["response"]
                    }
                    
                    if "accuracy_cosine" in result:
                        message_content["accuracy_cosine"] = result["accuracy_cosine"]
                    
                    if show_metrics:
                        metrics = evaluator.evaluate_response(
                            result["response"],
                            reference_answer=reference_answer.strip() if reference_answer.strip() else None
                        )
                        message_content["metrics"] = metrics
                    
                    st.session_state.messages.append(message_content)
                        
            elif model_choice == "Frontier (GPT-4o)" and frontier_ok:
                if reference_answer.strip():
                    # Use regular generate for accuracy calculation
                    with st.spinner("Generating..."):
                        result = frontier_model.generate(
                            prompt, 
                            subject=subject,
                            reference_answer=reference_answer.strip()
                        )
                        st.markdown(result["response"])
                        
                        # Show accuracy if available
                        if "accuracy_cosine" in result:
                            st.metric(
                                "Accuracy (Cosine)", 
                                f"{result['accuracy_cosine']:.3f}",
                                help="Cosine similarity with reference answer"
                            )
                        
                        # Prepare message with accuracy
                        message_content = {
                            "role": "assistant",
                            "content": result["response"]
                        }
                        
                        if "accuracy_cosine" in result:
                            message_content["accuracy_cosine"] = result["accuracy_cosine"]
                        
                        if show_metrics:
                            metrics = evaluator.evaluate_response(
                                result["response"],
                                reference_answer=reference_answer.strip()
                            )
                            message_content["metrics"] = metrics
                            with st.expander("Evaluation Metrics"):
                                st.json(metrics)
                        
                        st.session_state.messages.append(message_content)
                else:
                    # Use streaming for better UX when no reference answer
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for token in frontier_model.stream_generate(prompt, subject=subject):
                        full_response += token
                        response_placeholder.markdown(full_response)
                    
                    message_content = {
                        "role": "assistant",
                        "content": full_response
                    }
                    
                    if show_metrics:
                        metrics = evaluator.evaluate_response(full_response)
                        with st.expander("Evaluation Metrics"):
                            st.json(metrics)
                        message_content["metrics"] = metrics
                    
                    st.session_state.messages.append(message_content)
            else:
                st.error("Selected model is not available")

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
### How to Use
1. Select a model (Baseline or Frontier)
2. Choose your subject
3. Ask your O-Level question
4. Get step-by-step guidance!

**Remember:** This tutor guides you through problems rather than giving direct answers.
""")