import os
import random
import streamlit as st

from src.config import DEFAULT_TOP_K, ETHICS_POLICY
from src.rag.retriever import ChromaRetriever
from src.rag.llm import OllamaLLM


st.set_page_config(page_title="MedSimuli – Virtual Patient", page_icon="🩺", layout="wide")
st.title("MedSimuli: Virtual Patient (RAG + Ollama)")
st.caption("Prototype for demonstration only. Not medical advice.")

if "history" not in st.session_state:
    st.session_state.history = []
if "mode" not in st.session_state:
    st.session_state.mode = "Study (RAG QA)"
if "patient_pmc_id" not in st.session_state:
    st.session_state.patient_pmc_id = None
if "patient_persona" not in st.session_state:
    st.session_state.patient_persona = None

with st.sidebar:
    st.header("Settings")
    st.session_state.mode = st.selectbox(
        "Mode",
        ["Study (RAG QA)", "Virtual Patient"],
        index=0 if st.session_state.mode == "Study (RAG QA)" else 1,
    )
    top_k = st.slider("Top K", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    if st.session_state.mode == "Virtual Patient":
        st.markdown("---")
        st.subheader("Virtual Patient")
        current = st.session_state.patient_pmc_id or "None"
        st.text(f"Current case: {current}")
        if st.button("New patient"):
            retriever = ChromaRetriever()
            pmc = retriever.sample_pmc_id()
            st.session_state.patient_pmc_id = pmc
            # simple synthetic persona
            st.session_state.patient_persona = {
                "name": random.choice(["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan"]),
                "sex": random.choice(["male", "female"]),
                "age": random.choice([22, 35, 47, 60, 72]),
                "notes": "Cooperative, answers succinctly."
            }
            if pmc:
                contexts = retriever.retrieve("chief complaint presenting symptoms", top_k=top_k, pmc_id=pmc)
            else:
                contexts = []
            llm = OllamaLLM()
            opening = llm.generate_patient_reply("What brings you in today?", contexts, st.session_state.patient_persona)
            st.session_state.history.append({"role": "assistant", "content": opening, "contexts": contexts, "pmc_id": pmc})

    st.markdown("---")
    st.subheader("Index")
    if st.button("Show index path"):
        from src.config import PERSIST_DIR
        st.info(f"Chroma path: {os.path.abspath(PERSIST_DIR)}")

    st.markdown("---")
    with st.expander("Ethics/HIPAA policy in effect"):
        st.write(ETHICS_POLICY)

placeholder = "Ask about a case, symptoms, labs, or differential…" if st.session_state.mode == "Study (RAG QA)" else "Ask the patient a question…"
prompt = st.chat_input(placeholder)

if prompt:
    retriever = ChromaRetriever()
    llm = OllamaLLM()
    if st.session_state.mode == "Virtual Patient":
        pmc = st.session_state.patient_pmc_id
        contexts = retriever.retrieve(prompt, top_k=top_k, pmc_id=pmc)
        reply = llm.generate_patient_reply(prompt, contexts, st.session_state.patient_persona)
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": reply, "contexts": contexts, "pmc_id": pmc})
    else:
        contexts = retriever.retrieve(prompt, top_k=top_k)
        answer = llm.generate(prompt, contexts)
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": answer, "contexts": contexts})

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"]) 
        if turn.get("contexts"):
            with st.expander("Sources"):
                for c in turn["contexts"]:
                    st.markdown(f"- **[PMC_id]**: {c.get('pmc_id')} (chunk {c.get('chunk_index')}, score {c.get('score'):.3f})")
                    st.write(c.get("text"))
                    st.markdown("---")

st.markdown("---")
st.caption("Dataset: https://huggingface.co/datasets/chaoyi-wu/PMC-CaseReport | LLM: Ollama llama3.2:3b")
