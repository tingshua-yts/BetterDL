import nvidia.dali as dali
import nvidia.dali.types as types
import os


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="./model_repository/dali/1/model.dali")
    return parser.parse_args()


@dali.pipeline_def(batch_size=256, num_threads=4, device_id=0)
def pipe():
    images = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
    images = dali.fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = dali.fn.resize(images, resize_x=224, resize_y=224)
    images = dali.fn.crop_mirror_normalize(images,
                                           dtype=types.FLOAT,
                                           output_layout="CHW",
                                           crop=(224, 224),
                                           mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                           std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return images


def main(filename):
    pipe().serialize(filename=filename)
    print("Saved {}".format(filename))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    main(args.save)