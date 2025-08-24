from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import torch.nn as nn
import cv2
import numpy as np
import io
import base64
from PIL import Image, ImageFilter, ImageEnhance
import os
from typing import List
import uuid
import logging
import magic  
from pathlib import Path
import imghdr
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduced limits for memory efficiency
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB (reduced from 10 MB)
MAX_BATCH_SIZE = 10  # Reduced from 25
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/tiff'}

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Conv2d(256, 3, 9, padding=4)

    def forward(self, x):
        x = self.init(x)
        res = self.res_blocks(x)
        x = self.mid_conv(res)
        x = x + res
        x = self.upsample(x)
        return torch.tanh(self.output(x))

def validate_file_extension(filename: str) -> bool:
    """Validate file extension"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def validate_file_size(file_size: int) -> bool:
    """Validate file size"""
    return file_size <= MAX_FILE_SIZE

def validate_image_content(file_content: bytes, filename: str) -> bool:
    """Validate that file content is actually an image"""
    try:
        mime_type = magic.from_buffer(file_content, mime=True)
        if mime_type not in ALLOWED_MIME_TYPES:
            logger.warning(f"Invalid MIME type for {filename}: {mime_type}")
            return False
        
        image_type = imghdr.what(None, h=file_content)
        if image_type is None:
            logger.warning(f"Could not determine image type for {filename}")
            return False
        
        try:
            Image.open(io.BytesIO(file_content)).verify()
        except Exception as e:
            logger.warning(f"PIL verification failed for {filename}: {e}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating image content for {filename}: {e}")
        return False

async def validate_upload_file(file: UploadFile) -> tuple[bool, str]:
    """Comprehensive file validation"""
    if not file.filename:
        return False, "No filename provided"
    
    if not validate_file_extension(file.filename):
        return False, f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    if file.size and not validate_file_size(file.size):
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
    
    try:
        content = await file.read()
        await file.seek(0)
        
        if not validate_file_size(len(content)):
            return False, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
        
        if not validate_image_content(content, file.filename):
            return False, "File is not a valid image or has been corrupted"
        
        return True, "Valid"
        
    except Exception as e:
        logger.error(f"Error reading file {file.filename}: {e}")
        return False, f"Error reading file: {str(e)}"

def apply_natural_enhancement(image_array: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Apply natural-looking enhancement with crisp sharpening for SRGAN output
    """
    try:
        # Convert to PIL for initial processing
        pil_image = Image.fromarray(image_array)
        
        # 1. Enhanced sharpening for crisp pixels (more aggressive than before but still natural)
        sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
        enhanced = sharpness_enhancer.enhance(1 + strength * 0.8)  # Increased from 0.5
        
        # Convert to numpy for OpenCV sharpening
        cv_image = np.array(enhanced)
        
        # 2. Additional unsharp mask for pixel crispness
        gaussian_blur = cv2.GaussianBlur(cv_image, (0, 0), 0.8)
        unsharp_mask = cv2.addWeighted(cv_image, 1.5, gaussian_blur, -0.5, 0)
        
        # 3. Edge enhancement for pixel definition
        gray = cv2.cvtColor(unsharp_mask, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend edge enhancement for sharper pixels
        sharpened = cv2.addWeighted(unsharp_mask, 0.9, edges_colored, 0.1 * strength, 0)
        
        # Convert back to PIL for final touches
        pil_result = Image.fromarray(sharpened)
        
        # 4. Subtle contrast boost
        contrast_enhancer = ImageEnhance.Contrast(pil_result)
        pil_result = contrast_enhancer.enhance(1 + strength * 0.2)
        
        # 5. Minimal saturation boost for satellite imagery
        color_enhancer = ImageEnhance.Color(pil_result)
        pil_result = color_enhancer.enhance(1 + strength * 0.1)
        
        result = np.array(pil_result)
        
        # 6. Light noise reduction only if needed (preserve sharpness)
        if strength > 0.5:
            result = cv2.bilateralFilter(result, 3, 15, 15)  # Reduced parameters to preserve sharpness
        
        # Ensure values are in valid range
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Error in natural enhancement: {e}")
        return image_array

def get_memory_efficient_size(h: int, w: int, max_pixels: int = 1024*1024) -> tuple[int, int]:
    """Calculate memory-efficient image size"""
    total_pixels = h * w
    
    if total_pixels <= max_pixels:
        return h, w
    
    # Calculate scaling factor to stay under pixel limit
    scale_factor = np.sqrt(max_pixels / total_pixels)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Ensure dimensions are divisible by scale factor (4)
    new_h = (new_h // 4) * 4
    new_w = (new_w // 4) * 4
    
    # Ensure minimum size
    new_h = max(new_h, 64)
    new_w = max(new_w, 64)
    
    return new_h, new_w

app = FastAPI(title="SRGAN Satellite Image Enhancement API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
generator = None
MODEL_PATH = "generator_best.pth"
SCALE = 4
MAX_IMAGE_SIZE = 512  # Reduced from 2048 for memory efficiency

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_model():
    """Load the trained SRGAN generator model"""
    global generator
    try:
        generator = Generator().to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            generator.load_state_dict(state_dict)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        elif os.path.exists(MODEL_PATH.replace('.pth', '.pkl')):
            import pickle
            with open(MODEL_PATH.replace('.pth', '.pkl'), 'rb') as f:
                state_dict = pickle.load(f)
            generator.load_state_dict(state_dict)
            logger.info(f"Model loaded successfully from {MODEL_PATH.replace('.pth', '.pkl')}")
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        generator.eval()
        
        # Clear memory after loading
        clear_gpu_memory()
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def preprocess_image(image_array: np.ndarray) -> torch.Tensor:
    """Preprocess image for SRGAN model with memory optimization"""
    h, w = image_array.shape[:2]
    
    # Get memory-efficient size
    new_h, new_w = get_memory_efficient_size(h, w)
    
    if new_h != h or new_w != w:
        logger.info(f"Resizing from {h}x{w} to {new_h}x{new_w} for memory efficiency")
        image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Ensure dimensions are divisible by scale factor
    h, w = image_array.shape[:2]
    new_h = (h // SCALE) * SCALE
    new_w = (w // SCALE) * SCALE
    
    if new_h != h or new_w != w:
        image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create low-resolution version
    lr_h, lr_w = new_h // SCALE, new_w // SCALE
    lr_image = cv2.GaussianBlur(image_array, (3, 3), 0)  # Reduced blur
    lr_image = cv2.resize(lr_image, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    
    # Normalize to [-1, 1]
    lr_image = ((lr_image / 127.5) - 1).astype(np.float32)
    lr_tensor = torch.from_numpy(lr_image.transpose(2, 0, 1)).unsqueeze(0)
    
    return lr_tensor

def postprocess_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output back to numpy array"""
    image = (tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def numpy_to_bytes(image_array: np.ndarray, format: str = 'PNG', quality: int = 95) -> bytes:
    """Convert numpy array to bytes"""
    pil_image = Image.fromarray(image_array)
    img_buffer = io.BytesIO()
    
    if format.upper() == 'JPEG':
        pil_image.save(img_buffer, format=format, quality=quality, optimize=True)
    else:
        pil_image.save(img_buffer, format=format, optimize=True)
    
    img_buffer.seek(0)
    return img_buffer.getvalue()

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    os.makedirs("models", exist_ok=True)
    success = load_model()
    if not success:
        logger.warning("⚠️ Model not loaded - API will run in demo mode")

@app.get("/")
async def root():
    """Health check endpoint"""
    model_status = "loaded" if generator is not None else "not loaded"
    return {
        "message": "SRGAN Satellite Image Enhancement API v2.1 - Memory Optimized", 
        "device": DEVICE,
        "model_status": model_status,
        "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
        "max_batch_size": MAX_BATCH_SIZE,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "max_image_size": MAX_IMAGE_SIZE
    }

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    enhancement_level: str = "medium",  # low, medium, high
    apply_post_processing: bool = True
):
    """Memory-optimized single image processing with natural enhancement"""
    try:
        # Validate file
        is_valid, error_msg = await validate_upload_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"📸 Processing image: {file.filename} (size: {file.size} bytes)")
        
        # Clear GPU memory before processing
        clear_gpu_memory()
        
        # Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        original_size = image_array.shape[:2]
        
        if generator is None:
            # Enhanced demo mode with natural processing
            logger.warning("🔄 Running in demo mode with natural enhancement")
            # Simple bicubic upscaling
            target_h, target_w = original_size[0] * 2, original_size[1] * 2
            enhanced_array = cv2.resize(image_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            if apply_post_processing:
                strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
                strength = strength_map.get(enhancement_level, 0.5)
                enhanced_array = apply_natural_enhancement(enhanced_array, strength)
        else:
            # Real SRGAN processing with memory management
            logger.info(f"🖼️ Processing with SRGAN + natural {enhancement_level} enhancement")
            
            try:
                # Process with memory optimization
                lr_tensor = preprocess_image(image_array).to(DEVICE)
                
                with torch.no_grad():
                    sr_tensor = generator(lr_tensor)
                
                enhanced_array = postprocess_image(sr_tensor)
                
                # Clear GPU memory after processing
                del lr_tensor, sr_tensor
                clear_gpu_memory()
                
                # Apply enhanced natural post-processing for sharper pixels
                if apply_post_processing:
                    strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7}  # Increased for more sharpening
                    strength = strength_map.get(enhancement_level, 0.5)
                    enhanced_array = apply_natural_enhancement(enhanced_array, strength)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("GPU out of memory, falling back to CPU processing")
                    clear_gpu_memory()
                    
                    # Fallback to CPU or smaller processing
                    target_h, target_w = original_size[0] * 2, original_size[1] * 2
                    enhanced_array = cv2.resize(image_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    
                    if apply_post_processing:
                        strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
                        strength = strength_map.get(enhancement_level, 0.5)
                        enhanced_array = apply_natural_enhancement(enhanced_array, strength)
                else:
                    raise e
        
        # Convert to bytes
        enhanced_bytes = numpy_to_bytes(enhanced_array, quality=95)
        
        return StreamingResponse(
            io.BytesIO(enhanced_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=enhanced_4k_{file.filename}",
                "X-Original-Filename": file.filename,
                "X-Enhancement-Level": enhancement_level,
                "X-Post-Processing": str(apply_post_processing)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        clear_gpu_memory()  # Clear memory on error
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/enhance-batch")
async def enhance_batch(
    files: List[UploadFile] = File(...),
    enhancement_level: str = "medium",
    apply_post_processing: bool = True
):
    """Memory-optimized batch processing"""
    try:
        # Validate batch size
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many files. Maximum {MAX_BATCH_SIZE} images per batch"
            )
        
        # Validate each file
        valid_files = []
        validation_errors = []
        
        for i, file in enumerate(files):
            is_valid, error_msg = await validate_upload_file(file)
            if is_valid:
                valid_files.append(file)
            else:
                validation_errors.append(f"File {i+1} ({file.filename}): {error_msg}")
        
        if not valid_files:
            raise HTTPException(
                status_code=400, 
                detail=f"No valid files to process. Errors: {'; '.join(validation_errors)}"
            )
        
        import zipfile
        zip_buffer = io.BytesIO()
        
        logger.info(f"📄 Processing batch of {len(valid_files)} images")
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, file in enumerate(valid_files):
                try:
                    logger.info(f"📸 Processing {i+1}/{len(valid_files)}: {file.filename}")
                    
                    # Clear memory before each image
                    clear_gpu_memory()
                    
                    contents = await file.read()
                    image = Image.open(io.BytesIO(contents))
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image_array = np.array(image)
                    original_size = image_array.shape[:2]
                    
                    if generator is None:
                        target_h, target_w = original_size[0] * 2, original_size[1] * 2
                        enhanced_array = cv2.resize(image_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                        
                        if apply_post_processing:
                            strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
                            strength = strength_map.get(enhancement_level, 0.5)
                            enhanced_array = apply_natural_enhancement(enhanced_array, strength)
                    else:
                        try:
                            lr_tensor = preprocess_image(image_array).to(DEVICE)
                            with torch.no_grad():
                                sr_tensor = generator(lr_tensor)
                            enhanced_array = postprocess_image(sr_tensor)
                            
                            del lr_tensor, sr_tensor
                            clear_gpu_memory()
                            
                            if apply_post_processing:
                                strength_map = {"low": 0.2, "medium": 0.3, "high": 0.4}
                                strength = strength_map.get(enhancement_level, 0.3)
                                enhanced_array = apply_natural_enhancement(enhanced_array, strength)
                                
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                clear_gpu_memory()
                                # Fallback processing
                                target_h, target_w = original_size[0] * 2, original_size[1] * 2
                                enhanced_array = cv2.resize(image_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                                
                                if apply_post_processing:
                                    strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
                                    strength = strength_map.get(enhancement_level, 0.5)
                                    enhanced_array = apply_natural_enhancement(enhanced_array, strength)
                            else:
                                raise e
                    
                    enhanced_bytes = numpy_to_bytes(enhanced_array, quality=95)
                    zip_file.writestr(f"enhanced_4k_{file.filename}", enhanced_bytes)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {file.filename}: {e}")
                    error_msg = f"Error processing {file.filename}: {str(e)}"
                    zip_file.writestr(f"ERROR_{file.filename}.txt", error_msg.encode())
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.getvalue()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=enhanced_4k_batch_{len(valid_files)}_images.zip",
                "X-Processed-Count": str(len(valid_files)),
                "X-Total-Count": str(len(files)),
                "X-Enhancement-Level": enhancement_level
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing batch: {str(e)}")
        clear_gpu_memory()
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.post("/enhance-base64")
async def enhance_image_base64(data: dict):
    """Memory-optimized base64 processing"""
    try:
        image_data = data.get("image", "")
        enhancement_level = data.get("enhancement_level", "medium")
        apply_post_processing = data.get("apply_post_processing", True)
        
        if not image_data.startswith("data:image"):
            raise HTTPException(status_code=400, detail="Invalid base64 image format")
        
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        if not validate_file_size(len(image_bytes)):
            raise HTTPException(
                status_code=400, 
                detail=f"Image too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
            )
        
        if not validate_image_content(image_bytes, "base64_image"):
            raise HTTPException(status_code=400, detail="Invalid image content")
        
        clear_gpu_memory()
        
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        original_size = image_array.shape[:2]
        
        if generator is None:
            target_h, target_w = original_size[0] * 2, original_size[1] * 2
            enhanced_array = cv2.resize(image_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            if apply_post_processing:
                strength_map = {"low": 0.2, "medium": 0.3, "high": 0.4}
                strength = strength_map.get(enhancement_level, 0.3)
                enhanced_array = apply_natural_enhancement(enhanced_array, strength)
        else:
            try:
                lr_tensor = preprocess_image(image_array).to(DEVICE)
                with torch.no_grad():
                    sr_tensor = generator(lr_tensor)
                enhanced_array = postprocess_image(sr_tensor)
                
                del lr_tensor, sr_tensor
                clear_gpu_memory()
                
                if apply_post_processing:
                    strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
                    strength = strength_map.get(enhancement_level, 0.5)
                    enhanced_array = apply_natural_enhancement(enhanced_array, strength)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    clear_gpu_memory()
                    target_h, target_w = original_size[0] * 2, original_size[1] * 2
                    enhanced_array = cv2.resize(image_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    
                    if apply_post_processing:
                        strength_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
                        strength = strength_map.get(enhancement_level, 0.5)
                        enhanced_array = apply_natural_enhancement(enhanced_array, strength)
                else:
                    raise e
        
        enhanced_bytes = numpy_to_bytes(enhanced_array, quality=95)
        enhanced_b64 = base64.b64encode(enhanced_bytes).decode()
        
        return {
            "enhanced_image": f"data:image/png;base64,{enhanced_b64}",
            "status": "success",
            "enhancement_level": enhancement_level,
            "post_processing_applied": apply_post_processing
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        clear_gpu_memory()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the API"""
    gpu_memory = "N/A"
    if torch.cuda.is_available():
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory // 1024**2} MB"
    
    return {
        "model_loaded": generator is not None,
        "device": DEVICE,
        "gpu_memory": gpu_memory,
        "scale_factor": SCALE,
        "max_image_size": MAX_IMAGE_SIZE,
        "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
        "max_batch_size": MAX_BATCH_SIZE,
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "enhancement_levels": ["low", "medium", "high"],
        "post_processing": "Natural enhancement only"
    }

@app.get("/health")
async def health_check():
    """Health check with memory info"""
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() // 1024**2} MB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved() // 1024**2} MB"
        }
    
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": generator is not None,
        **memory_info
    }

@app.post("/clear-memory")
async def clear_memory_endpoint():
    """Manual memory clearing endpoint"""
    clear_gpu_memory()
    return {"status": "memory cleared"}

if __name__ == "__main__":
    import uvicorn
    
    os.makedirs("models", exist_ok=True)
    
    print("Starting Memory-Optimized SRGAN API v2.1...")
    print(f"Looking for model at: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Max file size: {MAX_FILE_SIZE // (1024*1024)} MB")
    print(f"Max batch size: {MAX_BATCH_SIZE} images")
    print(f"Max image size: {MAX_IMAGE_SIZE}px")
    print(f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )