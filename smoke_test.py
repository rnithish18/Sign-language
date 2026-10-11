import sys
import numpy as np

try:
    from src.predictor import GesturePredictor
    from src.landmark_utils import FrameResult, HandResult
    
    # Test ASL
    pred_asl = GesturePredictor("ASL")
    res_asl = pred_asl.update(np.random.rand(128).astype(np.float32))
    assert "smoothed_label" in res_asl
    print(f"ASL Predictor loaded and tested successfully. Prediction: {res_asl}")
    
    # Test ISL
    pred_isl = GesturePredictor("ISL")
    res_isl = pred_isl.update(np.random.rand(128).astype(np.float32))
    assert "smoothed_label" in res_isl
    print(f"ISL Predictor loaded and tested successfully. Prediction: {res_isl}")
    
    # Test cli_app imports
    import cli_app
    print("cli_app imported successfully.")
    
    # Test app.py imports
    try:
        import app
        print("app.py imported successfully.")
    except Exception as e:
        # Streamlit-webrtc raises NoSessionError when evaluated outside a Streamlit run context
        if "Failed to get the thread context" in str(e) or "NoSessionError" in type(e).__name__:
            print("app.py imported successfully (caught expected NoSessionError outside Streamlit context).")
        else:
            raise e
            
    print("Smoke test PASSED.")
    sys.exit(0)
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Smoke test FAILED: {e}")
    sys.exit(1)
