import pytest

from src.text_buffer import TextBuffer

def test_text_buffer_word_formation():
    buf = TextBuffer(auto_correct=False)
    
    # Type H E L L O
    for char in "HELLO":
        buf.handle_label(char)
        
    assert buf.current_word == "HELLO"
    assert buf.sentence == "HELLO"
    
    # Space commits the word
    buf.handle_label("SPACE")
    assert buf.current_word == ""
    assert buf.sentence == "HELLO"
    
def test_text_buffer_backspace():
    buf = TextBuffer(auto_correct=False)
    
    for char in "HELLO":
        buf.handle_label(char)
        
    buf.handle_label("BACKSPACE")
    assert buf.current_word == "HELL"
    
    # Backspace into previous word
    buf.handle_label("SPACE")
    buf.handle_label("BACKSPACE")
    assert buf.current_word == "HEL"
    assert buf.sentence == "HEL"
    
def test_text_buffer_clear():
    buf = TextBuffer(auto_correct=False)
    
    for char in "HI":
        buf.handle_label(char)
    buf.handle_label("SPACE")
    buf.handle_label("THERE")
    
    buf.handle_label("CLEAR")
    assert buf.current_word == ""
    assert buf.sentence == ""
    
def test_text_buffer_autocorrect():
    buf = TextBuffer(auto_correct=True)
    
    # Type 'HELLP' and space
    for char in "HELLP":
        buf.handle_label(char)
        
    buf.handle_label("SPACE")
    
    # Pyspellchecker should correct to 'HELLO' or similar. 
    # For robust test, we can just check if correction triggered.
    assert "->" in (buf.last_correction or "") or buf.finished_sentence[0].lower() != "hellp"

def test_text_buffer_speak():
    buf = TextBuffer(auto_correct=False)
    
    assert not buf.consume_speak_request()
    buf.handle_label("SPEAK")
    assert buf.speak_requested
    assert buf.consume_speak_request()
    assert not buf.consume_speak_request() # Should reset
