import os
from process_args import get_eval_args
from eval.evaluator import Evaluator


def evaluate_single_folder(folder_path, folder_name, base_args):
    """Evaluate a single folder"""
    print(f"\n{'='*50}")
    print(f"Evaluating folder: {folder_name}")
    print(f"{'='*50}")
    
    # Create a copy of args and modify the joints_folder
    import argparse
    args = argparse.Namespace(**vars(base_args))
    args.joints_folder = folder_path
    
    try:
        evaluator = Evaluator(args)
        if args.eval_mode == "loc":
            evaluator.evaluate_loc()
            return None, None  # loc mode doesn't return ADE/FDE
        elif args.eval_mode == "traj_pred":
            ade, fde = evaluator.evaluate_traj_pred()
            print(f"Successfully evaluated {folder_name}")
            return ade, fde
    except Exception as e:
        print(f"Error evaluating {folder_name}: {str(e)}")
        return None, None


def main():
    base_args = get_eval_args()
    
    # Hardcode the arguments for trajectory prediction evaluation
    import argparse
    base_args = argparse.Namespace()
    base_args.eval_mode = "traj_pred"
    base_args.load_traj = "../checkpoints/traj_pred/best_traj_model.pth"
    base_args.traj_cfg = "./configs/traj_pred.yaml"
    base_args.load_loc = ""  # Not needed for traj_pred mode
    base_args.loc_cfg = "./configs/localization.yaml"  # Default value
    base_args.obs = 4  # Default observation length
    base_args.pred = 6  # Default prediction length
    base_args.bs = 32  # Default batch size
    base_args.r_seed = 1  # Default random seed
    
    eval_mode = base_args.eval_mode
    assert eval_mode in ["loc", "traj_pred"]
    
    # Check if a specific joints_folder was provided via command line
    original_args = get_eval_args()
    if hasattr(original_args, 'joints_folder') and original_args.joints_folder != "./data/nusc_ped_data/":
        # If a specific folder was provided, evaluate only that folder
        print(f"Evaluating single folder: {original_args.joints_folder}")
        base_args.joints_folder = original_args.joints_folder
        evaluator = Evaluator(base_args)
        if eval_mode == "loc":
            evaluator.evaluate_loc()
        elif eval_mode == "traj_pred":
            ade, fde = evaluator.evaluate_traj_pred()
            print(f"\nSingle folder evaluation results:")
            print(f"ADE: {ade:.4f}")
            print(f"FDE: {fde:.4f}")
        return
    
    # Otherwise, evaluate all folders in the output directory
    output_dir = "../output/invited/"  # Relative to the code directory

    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        return
    
    # Get all subdirectories in the output folder
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    
    if not folders:
        print(f"No folders found in {output_dir}")
        return
    
    print(f"Found {len(folders)} folders to evaluate:")
    for folder in folders:
        print(f"  - {folder}")
    
    # Evaluate each folder
    successful_evaluations = 0
    failed_evaluations = 0
    ade_values = []
    fde_values = []
    successful_folders = []
    
    for folder_name in folders:
        folder_path = os.path.join(output_dir, folder_name)
        try:
            ade, fde = evaluate_single_folder(folder_path, folder_name, base_args)
            if ade is not None and fde is not None:
                ade_values.append(ade)
                fde_values.append(fde)
                successful_folders.append(folder_name)
                successful_evaluations += 1
            else:
                failed_evaluations += 1
        except Exception as e:
            print(f"Failed to evaluate {folder_name}: {str(e)}")
            failed_evaluations += 1
            continue
    
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total folders: {len(folders)}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations: {failed_evaluations}")
    
    if ade_values and fde_values:
        mean_ade = sum(ade_values) / len(ade_values)
        mean_fde = sum(fde_values) / len(fde_values)
        print(f"\nMETRICS ACROSS ALL FOLDERS:")
        print(f"Mean ADE: {mean_ade:.4f}")
        print(f"Mean FDE: {mean_fde:.4f}")
        print(f"\nIndividual folder results:")
        
        for i, folder_name in enumerate(successful_folders):
            print(f"  {folder_name}: ADE={ade_values[i]:.4f}, FDE={fde_values[i]:.4f}")
    else:
        print("No successful evaluations to calculate mean metrics.")
    
    print(f"{'='*50}")


if __name__ == '__main__':
    main()