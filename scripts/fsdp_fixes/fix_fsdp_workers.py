#!/usr/bin/env python3
"""
Fix fsdp_workers.py to properly load CustomSplitLLama models.
This script modifies the _build_model_optimizer method to detect and load CustomSplitLLama.
"""

import sys
import os
import re
from pathlib import Path


def fix_fsdp_workers(fsdp_workers_path: str):
    """Apply fixes to fsdp_workers.py"""

    print(f"Reading {fsdp_workers_path}...")
    with open(fsdp_workers_path, 'r') as f:
        content = f.read()

    # Backup original file
    backup_path = fsdp_workers_path + '.backup'
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_path}")

    # Fix 1: Update the import section
    old_import = '''################# CustomSplitLLama Registry #################
from transformers import LlamaConfig
from transformers import AutoConfig, AutoModelForCausalLM
from verl.models.custom_split_llama.modelling_custom_split_llama import CustomSplitLLamaForCausalLM

# AutoModelForCausalLM.register(LlamaConfig, CustomSplitLLamaForCausalLM)

# from vllm.model_executor.models import ModelRegistry
# ModelRegistry.register_model("CustomSplitLLamaForCausalLM", CustomSplitLLamaForCausalLM)
################# END of CustomSplitLLama Registry #################'''

    new_import = '''################# CustomSplitLLama Support #################
# Import CustomSplitLLama model
CUSTOM_SPLIT_LLAMA_AVAILABLE = False
CustomSplitLLamaForCausalLM = None

try:
    from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM
    CUSTOM_SPLIT_LLAMA_AVAILABLE = True
    logger.info("✅ CustomSplitLLama support enabled (verl.models.transformers)")
except ImportError:
    try:
        from verl.models.custom_split_llama.modelling_custom_split_llama import CustomSplitLLamaForCausalLM
        CUSTOM_SPLIT_LLAMA_AVAILABLE = True
        logger.info("✅ CustomSplitLLama support enabled (alternative import path)")
    except ImportError:
        logger.warning("⚠️  CustomSplitLLama not available - custom model support disabled")
        logger.warning("    Make sure custom_split_llama.py is in verl/models/transformers/ or verl/models/custom_split_llama/")
################# END CustomSplitLLama Support #################'''

    if old_import in content:
        content = content.replace(old_import, new_import)
        print("✅ Updated import section")
    else:
        print("⚠️  Could not find exact import section - it may have been modified")
        print("    Adding import at the beginning of the file...")
        # Find the line with device_name = get_device_name()
        device_name_match = re.search(r'(device_name = get_device_name\(\).*?\n)', content)
        if device_name_match:
            insert_pos = device_name_match.end()
            content = content[:insert_pos] + '\n' + new_import + '\n' + content[insert_pos:]
            print("✅ Added import section after device_name initialization")

    # Fix 2: Update the actor_module loading logic
    # Find the section where actor_module is loaded
    pattern = r'(print\(f"\[DEBUG\] actor_module_class : \{actor_module_class\}"\)\s*\n\s*actor_module = actor_module_class\.from_pretrained\()'

    replacement = r'''print(f"[DEBUG] actor_module_class : {actor_module_class}")

            # Check if this is a CustomSplitLLama model
            is_custom_split = (
                hasattr(actor_model_config, "architectures")
                and actor_model_config.architectures
                and len(actor_model_config.architectures) > 0
                and "CustomSplitLLamaForCausalLM" in actor_model_config.architectures[0]
            )

            print(f"[DEBUG] is_custom_split: {is_custom_split}")
            print(f"[DEBUG] CUSTOM_SPLIT_LLAMA_AVAILABLE: {CUSTOM_SPLIT_LLAMA_AVAILABLE}")

            if is_custom_split:
                if CUSTOM_SPLIT_LLAMA_AVAILABLE:
                    print(f"[DEBUG] Loading CustomSplitLLama from {local_path}")
                    actor_module = CustomSplitLLamaForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=local_path,
                        torch_dtype=torch_dtype,
                        config=actor_model_config,
                        trust_remote_code=trust_remote_code,
                    )
                else:
                    raise ImportError(
                        "CustomSplitLLamaForCausalLM architecture detected in config.json but "
                        "CustomSplitLLama model class is not available. "
                        "Please ensure custom_split_llama.py is installed in verl/models/transformers/ "
                        "or verl/models/custom_split_llama/"
                    )
            else:
                print(f"[DEBUG] Loading standard model with AutoModelForCausalLM")
                actor_module = actor_module_class.from_pretrained('''

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✅ Updated actor_module loading logic")
    else:
        print("❌ Could not find actor_module loading section")
        print("    You may need to manually add the CustomSplitLLama check")

    # Write the modified content
    with open(fsdp_workers_path, 'w') as f:
        f.write(content)

    print(f"✅ Successfully updated {fsdp_workers_path}")
    print(f"   Backup saved to: {backup_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_fsdp_workers.py <path_to_fsdp_workers.py>")
        print("\nExample:")
        print("  python fix_fsdp_workers.py /home/jovyan/rl/verl/verl/workers/fsdp_workers.py")
        sys.exit(1)

    fsdp_workers_path = sys.argv[1]

    if not os.path.exists(fsdp_workers_path):
        print(f"❌ Error: File not found: {fsdp_workers_path}")
        sys.exit(1)

    if not fsdp_workers_path.endswith('fsdp_workers.py'):
        print(f"⚠️  Warning: File doesn't appear to be fsdp_workers.py")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    try:
        fix_fsdp_workers(fsdp_workers_path)
        print("\n" + "="*60)
        print("✅ Fix applied successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Copy your custom_split_llama.py to verl/models/transformers/:")
        verl_dir = str(Path(fsdp_workers_path).parent.parent)
        print(f"   cp modelling_custom_split_llama.py {verl_dir}/models/transformers/custom_split_llama.py")
        print("\n2. Run your training script:")
        print("   bash grpo.sh")
    except Exception as e:
        print(f"\n❌ Error applying fix: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
