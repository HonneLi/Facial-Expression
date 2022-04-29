import torch
import torchvision
from torchvision import transforms
from torchvision.datasets.folder import make_dataset
from torch.utils.data import DataLoader
import mediapipe as mp
import cv2
import itertools


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def get_samples(root, extensions=(".mp4")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)
  
class RandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=60):
        super(RandomDataset).__init__()
        
        # Read all videos in the root file
        self.samples = get_samples(root)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            path, target = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer

            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
            start = 0
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
                current_pts = frame['pts']
            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts}
            yield output
            
            
transform = [transforms.Resize((112, 112))]
frame_transform = transforms.Compose(transform)

dataset = RandomDataset("./dataset", epoch_size=None, frame_transform=frame_transform)
loader = DataLoader(dataset, batch_size=6)
data = {"video": [], 'start': [], 'end': [], 'tensorsize': []}
for batch in loader:
    for i in range(len(batch['path'])):
        data['video'].append(batch['path'][i])
        data['start'].append(batch['start'][i].item())
        data['end'].append(batch['end'][i].item())
        data['tensorsize'].append(batch['video'][i].size())
print(data)
