import argparse
import pickle
import os

from haiku.data_structures import to_mutable_dict


def debug_print(data):
    print(type(data))
    print(data.keys())
    for k, v in data.items():
        head = "\t 1st for"
        print(head, k, type(k), type(v))
        attribute_list = list(dir(v))
        print(head, attribute_list)
        # param_key = "online_params"
        # if param_key in attribute_list:
        #     print(v.online_params == getattr(v, param_key))
        #     print(v.online_params.keys())
        if hasattr(v, "_asdict"):
            # BYOLState, _ByolExperimentState is subclass of namedtuple
            tmp_v_dict = v._asdict()
            print(head, k, type(tmp_v_dict), tmp_v_dict.keys())
        for ak in attribute_list:
            head2 = "\t\t 2nd for"
            tmp_attr = getattr(v, ak)
            # print(head2, "not dict", ak, type(tmp_attr))
            if ak == "__class__":
                continue
            if hasattr(tmp_attr, "items"):
                print(head2, ak, type(tmp_attr), len(tmp_attr.keys()))
            if hasattr(tmp_attr, "_asdict"):
                # ScaleByLarsState is subclass of namedtuple
                tmp_attr_dict = tmp_attr._asdict()
                print(head2, ak, type(tmp_attr_dict), tmp_attr_dict.keys())


def change_to_dict(haiku_dict):
    haiku_dict = to_mutable_dict(haiku_dict)
    for k, v in haiku_dict.items():
        if hasattr(v, "items") and not isinstance(v, dict):
            haiku_dict[k] = change_to_dict(v)
    return haiku_dict


def change_state_dict(byol_model_state_dict):
    out = {}
    for k, v in byol_model_state_dict.items():
        is_haiku = hasattr(v, "items") and not isinstance(v, dict)
        is_namedtuple = hasattr(v, "_asdict")
        out[k] = v
        if is_haiku:
            out[k] = change_to_dict(v)
        if is_namedtuple:
            tmp_dict = v._asdict()
            out[k] = change_state_dict(tmp_dict)
    return out


def main(input_pkl_filename, output_pkl_filename, is_debug_print=False):
    input_pkl_abs_path = os.path.abspath(input_pkl_filename)
    print("load", input_pkl_abs_path)
    with open(input_pkl_abs_path, "rb") as f:
        data = pickle.load(f)
    if is_debug_print:
        debug_print(data)
    data_keys = list(data.keys())
    byol_state_namedtuple = data[data_keys[0]]
    byol_model_state_dict = byol_state_namedtuple._asdict()
    model_state_dict = change_state_dict(byol_model_state_dict)
    data[data_keys[0]] = model_state_dict
    print(type(model_state_dict), model_state_dict.keys())
    for k, v in data.items():
        print(k, type(v))
    if is_debug_print:
        debug_print(model_state_dict)

    output_pkl_abs_path = os.path.abspath(output_pkl_filename)
    print("save to", output_pkl_abs_path)
    with open(output_pkl_abs_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", default="./pretrain.pkl", type=str)
    parser.add_argument("--output_pkl", default="./pretrain_after.pkl", type=str)
    parser.add_argument("--debug_print", action="store_true")
    args = parser.parse_args()
    main(args.input_pkl, args.output_pkl, args.debug_print)
