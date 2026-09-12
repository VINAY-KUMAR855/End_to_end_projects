from data.tusimple_loader import get_train_loader

loader = get_train_loader(root_dir="/kaggle/input/datasets/manideep1108/tusimple/TUSimple/train_set")
images, targets = next(iter(loader))

print("Images:",images.shape)

print("Lane existence:",targets["lane_exist"].shape)

print("Vertex existence:",targets["vertex_exist"].shape)

print("Lane x:",targets["lane_x"].shape)