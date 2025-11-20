import json
import zipfile
import os
import sys
import glob
from pathlib import Path
import numpy as np

def sse_print(event: str, data: dict) -> str:
    """
    SSE print
    :param event: Event name
    :param data: Event data (dict or object that can be serialized to JSON)
    :return: SSE format string
    """
    # Convert data to JSON string
    json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # Format according to SSE protocol
    message = f"event: {event}\n" \
              f"data: {json_str}\n"
    print(message, flush=True)


def sse_input_path_validated(args):
    try:
        if os.path.exists(args.input_path):
            event = "input_path_validated"
            data = {
                "status": "success",
                "message": "Input path is valid and complete.",
                "file_name": args.input_path
            }
            sse_print(event, data)
            
            try:
                if os.path.exists(f'{args.input_path}/data'):
                    event = "input_data_validated"
                    data = {
                        "status": "success",
                        "message": "Input data file is valid and complete.",
                        "file_name": glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
                    }
                    sse_print(event, data)
                else:
                    raise ValueError('Input data file not found.')
            except Exception as e:
                event = "input_data_validated"
                data = {
                    "status": "failure",
                    "message": f"{e}"
                }
                sse_print(event, data)
                
            try:
                if os.path.exists(f'{args.input_path}/model'):
                    event = "input_model_validated"
                    data = {
                        "status": "success",
                        "message": "Input model file is valid and complete.",
                        "file_name": glob.glob(os.path.join(f'{args.input_path}/model', '*'))[0]
                    }
                    sse_print(event, data)
                else:
                    raise ValueError('Input model file not found.')
            except Exception as e:
                event = "input_model_validated"
                data = {
                    "status": "failure",
                    "message": f"{e}"
                }
                sse_print(event, data)
        else:
            raise ValueError('Input path not found.')
    except Exception as e:
        event = "input_path_validated"
        data = {
            "status": "failure",
            "message": f"{e}"
        }
        sse_print(event, data)


def sse_output_path_validated(args):
    try:
        if os.path.exists(args.output_path):
            event = "output_path_validated"
            data = {
                "status": "success",
                "message": "Output path is valid and complete.",
                "file_name": args.output_path
            }
            sse_print(event, data)
        else:
            raise ValueError('Output path not found.')
    except Exception as e:
        event = "output_path_validated"
        data = {
            "status": "failure",
            "message": f"{e}"
        }
        sse_print(event, data)
            

def sse_adv_samples_gen_validated(adv_image_name):
    event = "adv_samples_gen_validated"
    data = {
        "status": "success",
        "message": "adv sample is generated.",
        "file_name": adv_image_name
    }
    sse_print(event, data)


def sse_clean_samples_gen_validated(clean_image_name):
    event = "clean_samples_gen_validated"
    data = {
        "status": "success",
        "message": "clean sample is generated.",
        "file_name": clean_image_name
    }
    sse_print(event, data)


def sse_epoch_progress(progress, total, epoch_type="Epoch"):
    """
    Output SSE format for epoch progress
    :param progress: Current epoch number
    :param total: Total epochs
    :param epoch_type: Type of progress (default: "Epoch")
    """
    event = "training_progress"
    data = {
        "progress": progress,
        "total": total,
        "type": epoch_type
    }
    sse_print(event, data)


def sse_error(message, event_name="error"):
    """
    Output SSE format for error messages
    :param message: Error message
    :param event_name: Event name (default: "error")
    """
    data = {
        "status": "failure",
        "message": message
    }
    sse_print(event_name, data)


def sse_class_number_validation(expected, got):
    """
    Output SSE format for CLASS_NUMBER mismatch error
    :param expected: Expected class number
    :param got: Got class number
    """
    event = "class_number_validated"
    data = {
        "status": "failure",
        "message": f"expect CLASS_NUMBER {expected} but got {got}"
    }
    sse_print(event, data)


def sse_final_result(results: dict, event_name="final_result"):
    """
    Output SSE format for final results
    :param results: Results dictionary
    :param event_name: Event name (default: "final_result")
    """
    sse_print(event_name, results)

