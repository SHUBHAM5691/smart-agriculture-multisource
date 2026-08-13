import streamlit as st


def initialize_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("prediction", None)


def clear_session() -> None:
    st.session_state.clear()
    initialize_session()


def store_prediction(prediction: dict) -> None:
    st.session_state["prediction"] = prediction


def add_message(role: str, content: str) -> None:
    st.session_state["messages"].append({"role": role, "content": content})


def get_prediction() -> dict | None:
    return st.session_state["prediction"]


def get_memory() -> dict:
    return {
        "messages": st.session_state["messages"],
        "prediction": st.session_state["prediction"],
    }
