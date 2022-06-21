import argparse

def parse_opt():
    print("--- Entering parse_opt")
    parser = argparse.ArgumentParser(description="Test of argparse")
    parser.add_argument("-text", type=str)
    parser.add_argument("--text2", type=str)
    parser.add_argument("number", type=int, default=1)
    opt = parser.parse_args()
    print(f"---- type(opt): {type(opt)}")
    print(f"---- opt.number: {opt.number}")
    print(f"---- opt.text: {opt.text}")
    print(f"---- opt.text2: {opt.text2}")
    return opt

def run(text="Std Text",
        text2="Std Text 2",
        number=1, 
        ):
    print("--- Entering run")
    print(f"--- text: {text}")
    print(f"--- text2: {text2}")
    print(f"--- number: {number}")
    
    for i in range(number):
        print(text)
    print(text2)
    

def main(opt):
    print("--- Entering Main")
    print(dict(**vars(opt)))
    run(**vars(opt))

if __name__ == "__main__":
    print("--- Entered if")
    opt = parse_opt()
    main(opt)