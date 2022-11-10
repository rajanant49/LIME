import os
import cv2
from LIME import LIME
import argparse
from argparse import RawTextHelpFormatter

def main(args):
    # Initialize the LIME operator and load the image
    lime = LIME(iter=args.iterations,
                alpha=args.alpha,
                rho=args.rho,
                gamma=args.gamma,
                strategy=args.strategy, 
                eps=args.epsilon,
                sigma=args.sigma)

    opencv_image = cv2.imread(args.filePath)
    lime.loadimage(opencv_image)
    #run
    R = lime.run()
    #save picture
    filename = os.path.split(args.filePath)[-1]
    savePath = f"./pics/LIME_{filename}"
    cv2.imwrite(savePath, R)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python implementation of low-light image enhancement (LIME) techniques via illumination map estimation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-f", '--filePath', default='./pics/2.jpg', type=str,
                        help="number of iterations to converge")
    parser.add_argument("-it", '--iterations', default=30, type=int,
                        help="number of iterations to converge")
    parser.add_argument("-alph", '--alpha', default=0.15, type=float,
                        help="the alpha balancing parameter")
    parser.add_argument("-rho", '--rho', default=1.1, type=float,
                        help="scaling factor for miu")
    parser.add_argument("-mu", "--miu", default=0.15, type=float,
                        help="positive penalty scalar.")
    parser.add_argument("-gm", '--gamma', default=0.6, type=float,
                        help=" the gamma correction factor")
    parser.add_argument("-st", "--strategy", default=2, type=int,
                        help="1: Strategy 1, 2: Strategy 2")
    parser.add_argument("-eps", "--epsilon", default=0.1, type=float,
                        help="constant to avoid computation instability.")
    parser.add_argument("-s", '--sigma', default=3, type=int,
                        help="Spatial standard deviation for spatial affinity based Gaussian weights.")

    args = parser.parse_args()
    # print(args)
    main(args)
