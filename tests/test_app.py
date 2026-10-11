from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_without_api_key():
    app_path = (
        Path(__file__).resolve().parents[1]
        / 'src'
        / 'corporate_rag'
        / 'app.py'
    )

    app = AppTest.from_file(str(app_path)).run()

    assert not app.exception
    assert app.title[0].value == '📚 Corporate Knowledge Assistant'
    assert len(app.chat_input) == 1