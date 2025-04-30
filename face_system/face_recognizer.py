import torch
import torch.nn.functional as F


class FaceRecognizer:
    def __init__(self, model_name, device, amp_enabled):
        self.device = device
        self.amp_enabled = amp_enabled
        self.model = torch.hub.load(
            repo_or_dir="otroshi/edgeface",
            model=model_name,
            source="github",
            pretrained=True,
        ).to(device)

    def recognize_batch(self, aligned_faces):
        batch = torch.stack(aligned_faces).to(self.device)

        if self.amp_enabled:
            batch = batch.half()

        with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            embeddings = self.model(batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().float().cpu().numpy()
