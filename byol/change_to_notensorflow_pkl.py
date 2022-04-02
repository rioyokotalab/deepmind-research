import argparse
import glob
import pickle
import os
import time

from haiku.data_structures import to_mutable_dict


def debug_print(data, head_msg=""):
    if head_msg != "":
        print(head_msg)
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
        if type(v) == dict:
            for lk, lv in v.items():
                head2 = "\t\t 2nd for dict"
                print(head2, lk, type(lv))
                if type(lv) == list:
                    head3 = "\t\t\t 3rd for list"
                    for li in lv:
                        print(head3, lk, type(li))


def my_is_file(filename, debug=False):
    splits = filename.split(".", 1)
    is_file = len(splits) > 1
    if debug:
        print(splits)
        if is_file:
            basename_without_ext, ext = splits
            print(basename_without_ext, ext)
    return is_file


def change_haiku_to_dict(haiku_dict):
    haiku_dict = to_mutable_dict(haiku_dict)
    for k, v in haiku_dict.items():
        if hasattr(v, "items") and not isinstance(v, dict):
            haiku_dict[k] = change_haiku_to_dict(v)
    return haiku_dict


def change_state_dict(byol_model_state_dict):
    is_dict = isinstance(byol_model_state_dict, dict)
    is_haiku = hasattr(byol_model_state_dict, "items") and not is_dict
    is_namedtuple = hasattr(byol_model_state_dict, "_asdict")
    is_list = isinstance(byol_model_state_dict, list)
    is_tuple = isinstance(byol_model_state_dict, tuple)
    if is_haiku:
        return change_haiku_to_dict(byol_model_state_dict)
    if is_namedtuple:
        tmp_dict = byol_model_state_dict._asdict()
        return change_state_dict(tmp_dict)
    if is_list or is_tuple:
        tmp_list = []
        for v in byol_model_state_dict:
            tmp = change_state_dict(v)
            tmp_list.append(tmp)
        return tmp_list
    if is_dict:
        for k, v in byol_model_state_dict.items():
            byol_model_state_dict[k] = change_state_dict(v)
    return byol_model_state_dict


def change_single_pkl(input_pkl_filename, output_pkl_filename, is_debug_print=False):
    s_time = time.perf_counter()

    input_pkl_abs_path = os.path.abspath(input_pkl_filename)
    print("load", input_pkl_filename)
    with open(input_pkl_abs_path, "rb") as f:
        data = pickle.load(f)
    if is_debug_print:
        debug_print(data, head_msg="before change data")
    data = change_state_dict(data)
    model_state_dict = data[list(data.keys())[0]]
    if is_debug_print:
        print(type(model_state_dict), model_state_dict.keys())
        debug_print(data, head_msg="after change data")
        debug_print(model_state_dict, head_msg="model state dict print")

    output_pkl_abs_path = os.path.abspath(output_pkl_filename)
    print("save to", output_pkl_filename)
    with open(output_pkl_abs_path, "wb") as f:
        pickle.dump(data, f)

    e_time = time.perf_counter()
    single_exec_time = e_time - s_time
    print_str = f"process for {input_pkl_filename} exec time: {single_exec_time}s"
    print(print_str)


def main(input_pkl, output_pkl, is_overwrite=False, is_debug_print=False):
    s_time = time.perf_counter()

    default_suffix = "_standard."

    is_exist_input = os.path.exists(input_pkl)
    msg = f'Not found file or dir: "{input_pkl}"'
    assert is_exist_input, msg

    is_input_file = my_is_file(input_pkl, debug=is_debug_print)
    is_output_file = my_is_file(output_pkl, debug=is_debug_print)
    msg = "should match input dir type and output dir type: "
    msg += f"input path: {input_pkl} output path: {output_pkl}"
    assert is_input_file or not is_output_file, msg

    if not is_output_file:
        os.makedirs(output_pkl, exist_ok=True)
    is_same_dir = input_pkl == output_pkl
    if is_input_file:
        tmp_input_abs_path = os.path.abspath(input_pkl)
        input_dir = os.path.dirname(tmp_input_abs_path)
        output_dir = os.path.abspath(output_pkl)
        if is_output_file:
            output_dir = os.path.dirname(output_dir)
        is_same_dir = input_dir == output_dir
    if is_same_dir and is_output_file and not is_overwrite:
        output_basename = os.path.basename(output_pkl)
        tmp_output_dir = os.path.dirname(output_pkl)
        output_basename_without_ext, ext = output_basename.split(".", 1)
        tmp_output_filename = output_basename_without_ext + default_suffix + ext
        output_pkl = os.path.join(tmp_output_dir, tmp_output_filename)

    input_files = [input_pkl]
    if not is_input_file:
        input_files = sorted(glob.glob(os.path.join(input_pkl, "*.pkl")))
    for input_file in input_files:
        output_pkl_filename = output_pkl
        if not is_output_file:
            input_basename = os.path.basename(input_file)
            if is_same_dir:
                basename_without_ext, ext = input_basename.split(".", 1)
                input_basename = basename_without_ext + default_suffix + ext
            output_pkl_filename = os.path.join(output_pkl, input_basename)
        change_single_pkl(input_file, output_pkl_filename, is_debug_print)

    e_time = time.perf_counter()
    total_e_time = e_time - s_time
    print_str = f"process for {input_pkl} total exec time: {total_e_time}s"
    print(print_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", default="./pretrain.pkl", type=str)
    parser.add_argument("--output_pkl", default="./pretrain_after.pkl", type=str)
    parser.add_argument("--overwrite_pklfile", action="store_true")
    parser.add_argument("--debug_print", action="store_true")
    args = parser.parse_args()
    main(args.input_pkl, args.output_pkl, args.overwrite_pklfile, args.debug_print)
