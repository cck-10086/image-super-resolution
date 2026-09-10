from flask import Flask, request, jsonify, send_from_directory, stream_with_context, Response
from flask_cors import CORS
import os
import uuid
from PIL import Image
import sys
import time
import logging
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SuperResolutionModel, bicubic_interpolation

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, 'frontend')

app = Flask(__name__,
            template_folder=os.path.join(FRONTEND_DIR, 'templates'),
            static_folder=os.path.join(FRONTEND_DIR, 'static'))
CORS(app)

UPLOAD_FOLDER = os.path.join(BACKEND_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BACKEND_DIR, 'outputs')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

model_x2 = None
model_x4 = None
device = None
load_time = 0

model_type = 'edsr'  # 'edsr' or 'realesrgan'

def load_models():
    global model_x2, model_x4, device, load_time, model_type
    start_time = time.time()
    logger.info("Loading super-resolution models...")

    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_x2 = SuperResolutionModel(scale_factor=2)

        # Check for Real-ESRGAN weights
        resrgan_path = os.path.join(PROJECT_DIR, 'model_weights', 'RealESRGAN_x4plus.pth')
        if os.path.exists(resrgan_path):
            logger.info("Real-ESRGAN weights found, using RRDBNet for 4x...")
            from model.rrdbnet import RRDBNet, RealESRGANWrapper
            rrbdnet = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
            checkpoint = torch.load(resrgan_path, map_location=device, weights_only=False)
            if 'params' in checkpoint:
                rrbdnet.load_state_dict(checkpoint['params'], strict=False)
            elif 'params_ema' in checkpoint:
                rrbdnet.load_state_dict(checkpoint['params_ema'], strict=False)
            else:
                rrbdnet.load_state_dict(checkpoint, strict=False)
            model_x4 = RealESRGANWrapper(rrbdnet.to(device).eval(), device)
            model_type = 'realesrgan'
            logger.info("Real-ESRGAN x4plus loaded!")
        else:
            logger.info("Loading EDSR x4 model (Real-ESRGAN weights not found)...")
            model_x4 = SuperResolutionModel(scale_factor=4)
            model_type = 'edsr'

        load_time = time.time() - start_time
        logger.info(f"Models loaded successfully on {device}!")
        logger.info(f"Load time: {load_time:.2f} seconds")
        logger.info(f"4x model type: {model_type}")

        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

@app.route('/')
def index():
    return send_from_directory(os.path.join(FRONTEND_DIR, 'templates'), 'index.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'static'), path)

@app.route('/api/enhance', methods=['POST'])
def enhance():
    start_time = time.time()

    if 'image' not in request.files:
        logger.warning("No image uploaded")
        return jsonify({'error': 'No image uploaded', 'code': 400}), 400

    file = request.files['image']
    scale_factor = int(request.form.get('scale', 2))
    method = request.form.get('method', 'edsr')

    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected', 'code': 400}), 400

    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']:
        logger.warning(f"Unsupported file type: {file_ext}")
        return jsonify({'error': f'Unsupported file type: {file_ext}', 'code': 400}), 400

    unique_id = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{file.filename}")

    try:
        file.save(original_path)
        logger.info(f"Uploaded image: {file.filename} -> {original_path}")

        original_img = Image.open(original_path)
        original_size = original_img.size
        logger.info(f"Original image size: {original_size}")

        if original_size[0] * original_size[1] > 4096 * 4096:
            logger.warning(f"Image too large: {original_size}")
            return jsonify({'error': 'Image size exceeds maximum limit (4096x4096)', 'code': 413}), 413

        process_start = time.time()

        if method == 'bicubic':
            sr_image = bicubic_interpolation(original_path, scale_factor)
            logger.info(f"Using bicubic interpolation, scale: {scale_factor}x")
        else:
            if scale_factor == 2:
                model = model_x2
            elif scale_factor == 4:
                model = model_x4
            elif scale_factor == 8:
                # 级联: 先4x再2x = 8x, 复用训练好的模型
                sr_image = model_x4.enhance(original_path)
                sr_image.save(original_path + '_temp.png')
                sr_image = model_x2.enhance(original_path + '_temp.png')
                import os as _os
                _os.remove(original_path + '_temp.png')
                logger.info(f"Using cascaded EDSR (4x+2x), scale: 8x")
                process_time = time.time() - process_start
                output_filename = f"sr_{unique_id}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                sr_image.save(output_path, 'PNG', quality=95, optimize=True)
                sr_size = sr_image.size
                total_time = time.time() - start_time
                logger.info(f"Processing completed: {original_size} -> {sr_size}")
                logger.info(f"Process time: {process_time:.2f}s, Total time: {total_time:.2f}s")
                return jsonify({
                    'success': True,
                    'original': {
                        'filename': file.filename,
                        'width': original_size[0],
                        'height': original_size[1],
                        'path': f'/uploads/{unique_id}_{file.filename}'
                    },
                    'enhanced': {
                        'width': sr_size[0],
                        'height': sr_size[1],
                        'scale': scale_factor,
                        'method': method,
                        'path': f'/outputs/{output_filename}'
                    },
                    'metrics': {
                        'process_time_ms': int(process_time * 1000),
                        'total_time_ms': int(total_time * 1000),
                        'device': device
                    }
                })
            else:
                logger.error(f"Invalid scale factor: {scale_factor}")
                return jsonify({'error': 'Invalid scale factor (must be 2, 4, or 8)', 'code': 400}), 400

            sr_image = model.enhance(original_path)
            logger.info(f"Using EDSR model, scale: {scale_factor}x")

        process_time = time.time() - process_start

        output_filename = f"sr_{unique_id}.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        sr_image.save(output_path, 'PNG', quality=95, optimize=True)

        sr_size = sr_image.size
        total_time = time.time() - start_time

        logger.info(f"Processing completed: {original_size} -> {sr_size}")
        logger.info(f"Process time: {process_time:.2f}s, Total time: {total_time:.2f}s")

        return jsonify({
            'success': True,
            'original': {
                'filename': file.filename,
                'width': original_size[0],
                'height': original_size[1],
                'path': f'/uploads/{unique_id}_{file.filename}'
            },
            'enhanced': {
                'width': sr_size[0],
                'height': sr_size[1],
                'scale': scale_factor,
                'method': method,
                'path': f'/outputs/{output_filename}'
            },
            'metrics': {
                'process_time_ms': int(process_time * 1000),
                'total_time_ms': int(total_time * 1000),
                'device': device
            }
        })

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return jsonify({'error': str(e), 'code': 500}), 500

@app.route('/api/enhance/stream', methods=['POST'])
def enhance_stream():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    scale_factor = int(request.form.get('scale', 2))
    method = request.form.get('method', 'edsr')

    @stream_with_context
    def generate():
        yield '{"status": "starting", "message": "Initializing..."}\n'

        try:
            unique_id = str(uuid.uuid4())
            original_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{file.filename}")
            file.save(original_path)

            yield '{"status": "processing", "message": "Loading image..."}\n'

            original_img = Image.open(original_path)

            yield '{"status": "processing", "message": "Running inference..."}\n'

            if method == 'bicubic':
                sr_image = bicubic_interpolation(original_path, scale_factor)
            else:
                if scale_factor == 2:
                    sr_image = model_x2.enhance(original_path)
                elif scale_factor == 4:
                    sr_image = model_x4.enhance(original_path)
                elif scale_factor == 8:
                    sr_image = model_x4.enhance(original_path)
                    sr_image.save(original_path + "_temp.png")
                    sr_image = model_x2.enhance(original_path + "_temp.png")
                    os.remove(original_path + "_temp.png")
                else:
                    raise ValueError(f"Unsupported scale factor: {scale_factor}")
            yield '{"status": "processing", "message": "Saving result..."}\n'

            output_filename = f"sr_{unique_id}.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            sr_image.save(output_path)

            yield f'{{"status": "completed", "message": "Done!", "output_path": "/outputs/{output_filename}"}}\n'

        except Exception as e:
            yield f'{{"status": "error", "message": "{str(e)}"}}\n'

    return Response(generate(), mimetype='application/json')

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'online',
        'device': device,
        'models_loaded': model_x2 is not None and model_x4 is not None,
        'load_time_ms': int(load_time * 1000),
        'scale_factors': [2, 4, 8],
        'methods': ['edsr', 'bicubic'],
        'model_type': model_type
    })

@app.route('/api/metrics')
def metrics():
    import psutil
    import torch

    metrics = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'device': device
    }

    if device == 'cuda':
        metrics.update({
            'cuda_memory_used': torch.cuda.memory_allocated() / (1024 ** 2),
            'cuda_memory_cached': torch.cuda.memory_reserved() / (1024 ** 2)
        })

    return jsonify(metrics)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<path:filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    try:
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        for f in os.listdir(OUTPUT_FOLDER):
            os.remove(os.path.join(OUTPUT_FOLDER, f))
        logger.info("Cache cleared")
        return jsonify({'success': True, 'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'code': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'code': 500}), 500

if __name__ == '__main__':
    load_models()
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
