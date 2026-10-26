from transformers import SamModel, SamProcessor


def build_finetunable_sam(device="mps"):
    print("Downloading SAM Processor and Model from Hugging Face...")
    # use the base SAM model (highly optimized for PyTorch)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    print("Freezing Image Encoder and Prompt Encoder...")
    # freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze only the mask decoder
    for name, param in model.mask_decoder.named_parameters():
        param.requires_grad = True

    # check trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Model built! Trainable parameters: {trainable_params:,}")

    return model, processor
