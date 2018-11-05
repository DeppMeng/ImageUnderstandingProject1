using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing.Imaging;

namespace HarrisDetectorCSharp
{
    class Program
    {
        static int[] GradientKernel = { -2, -1, 0, 1, 2 };
        static int iHeight = 0;
        static int iWidth = 0;
        static Tuple<double[,], double[,]> gradientB;
        static Tuple<double[,], double[,]> gradientG;
        static Tuple<double[,], double[,]> gradientR;
        static List<Tuple<int, int>> topklist;
        static bool parallel = true;

        static void Main(string[] args)
        {
            string str1 = "C:/Depp Data/Others/Wallpaper/Lord of the Ring/harristest_small1.jpg";
            //string str1 = "C:/Depp Data/Others/Wallpaper/Lord of the Ring/1.jpg";
            //string str1 = "C:/Users/mdp/OneDrive/Pictures/img1.jpg";

            Bitmap image1 = GetImage(str1);
            iHeight = image1.Height;
            iWidth = image1.Width;
            //RGB2Grey(image);
            int[,] responsemap = ANMSHarrisDetector(image1, 250);
            List<int[,]> finalmap = CombineHarrisValueImage(GetBGRList(image1), responsemap);
            VisualizeBGRList(finalmap, "small_test_anms_1");
            List<Tuple<int[], int, int>> sift_feature_image_1 = GetSIFTFeature(topklist);

            string str2 = "C:/Depp Data/Others/Wallpaper/Lord of the Ring/harristest_small2.jpg";

            Bitmap image2 = GetImage(str2);
            responsemap = ANMSHarrisDetector(image2, 250);
            finalmap = CombineHarrisValueImage(GetBGRList(image2), responsemap);
            VisualizeBGRList(finalmap, "small_test_anms_2");
            List<Tuple<int[], int, int>> sift_feature_image_2 = GetSIFTFeature(topklist);
            List<Tuple<int, int, int>> match_list = GetTop1Match(sift_feature_image_1, sift_feature_image_2);
            VisualizeMatch(GetBGRList(image1), GetBGRList(image2), match_list, 50, sift_feature_image_1, sift_feature_image_2);
        }
        // Get a Bitmap instance from a path string
        static Bitmap GetImage(string img)
        {

            Bitmap image;
            image = new Bitmap(img);
            return image;
        }
        // Convert RGB image to GreyScale image
        static void RGB2Grey(Bitmap bitmap)
        {
            Rectangle rect = new Rectangle(0, 0, iWidth, iHeight);
            BitmapData bmpData = bitmap.LockBits(rect,
                ImageLockMode.ReadWrite, bitmap.PixelFormat);

            IntPtr iPtr = bmpData.Scan0;
            int iBytes = iWidth * iHeight * 3;
            byte[] PixelValues = new byte[iBytes];
            byte[] newPixelValues = new byte[PixelValues.Length];
            System.Runtime.InteropServices.Marshal.Copy(iPtr, PixelValues, 0, iBytes);

            bitmap.UnlockBits(bmpData);
            int[,] R = new int[iHeight, iWidth];
            int[,] G = new int[iHeight, iWidth];
            int[,] B = new int[iHeight, iWidth];
            int[,] Grey = new int[iHeight, iWidth];

            int iPoint = 0;

            for (int i = 0; i < iHeight; i++)
            {
                for (int j = 0; j < iWidth; j++)
                {
                    B[i, j] = Convert.ToInt32(PixelValues[iPoint++]);
                    G[i, j] = Convert.ToInt32(PixelValues[iPoint++]);
                    R[i, j] = Convert.ToInt32(PixelValues[iPoint++]);
                    newPixelValues[iPoint - 3] = Convert.ToByte((B[i, j] + G[i, j] + R[i, j]) / 3);
                    newPixelValues[iPoint - 2] = Convert.ToByte((B[i, j] + G[i, j] + R[i, j]) / 3);
                    newPixelValues[iPoint - 1] = Convert.ToByte((B[i, j] + G[i, j] + R[i, j]) / 3);
                }
            }


            Bitmap Result = new Bitmap(iWidth, iHeight,
            PixelFormat.Format24bppRgb);
            Rectangle newrect = new Rectangle(0, 0, iWidth, iHeight);
            BitmapData newbmpData = Result.LockBits(newrect,
                ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            IntPtr newiPtr = newbmpData.Scan0;
            System.Runtime.InteropServices.Marshal.Copy(newPixelValues, 0, newiPtr, PixelValues.Length);
            Result.Save("C:/Depp Data/Others/Wallpaper/Lord of the Ring/1_grey.jpg", ImageFormat.Jpeg);
        }
        // Get the RGB pixel matrix from a Bitmap instance
        static List<int[,]> GetBGRList(Bitmap bitmap)
        {
            Rectangle rect = new Rectangle(0, 0, iWidth, iHeight);
            BitmapData bmpData = bitmap.LockBits(rect,
                ImageLockMode.ReadWrite, bitmap.PixelFormat);

            IntPtr iPtr = bmpData.Scan0;
            int iBytes = iWidth * iHeight * 3;
            byte[] PixelValues = new byte[iBytes];

            System.Runtime.InteropServices.Marshal.Copy(iPtr, PixelValues, 0, iBytes);

            bitmap.UnlockBits(bmpData);
            int[,] R = new int[iHeight, iWidth];
            int[,] G = new int[iHeight, iWidth];
            int[,] B = new int[iHeight, iWidth];

            int iPoint = 0;

            for (int i = 0; i < iHeight; i++)
            {
                for (int j = 0; j < iWidth; j++)
                {
                    B[i, j] = Convert.ToInt32(PixelValues[iPoint++]);
                    G[i, j] = Convert.ToInt32(PixelValues[iPoint++]);
                    R[i, j] = Convert.ToInt32(PixelValues[iPoint++]);
                }
            }

            List<int[,]> result = new List<int[,]>();
            result.Add(B);
            result.Add(G);
            result.Add(R);
            return result;
        }
        // Visualize an image from a RGB pixel matrix
        static void VisualizeBGRList(List<int[,]> list, string name)
        {
            int[,] B = list[0];
            int[,] G = list[1];
            int[,] R = list[2];
            byte[] PixelValues = new byte[iWidth * iHeight * 3];

            int iPoint = 0;

            for (int i = 0; i < iHeight; i++)
            {
                for (int j = 0; j < iWidth; j++)
                {
                    PixelValues[iPoint++] = Convert.ToByte(B[i, j]);
                    PixelValues[iPoint++] = Convert.ToByte(G[i, j]);
                    PixelValues[iPoint++] = Convert.ToByte(R[i, j]);
                }
            }

            Bitmap bitmap = new Bitmap(iWidth, iHeight, PixelFormat.Format24bppRgb);
            Rectangle newrect = new Rectangle(0, 0, iWidth, iHeight);
            BitmapData newbmpData = bitmap.LockBits(newrect,
                ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            IntPtr newiPtr = newbmpData.Scan0;
            System.Runtime.InteropServices.Marshal.Copy(PixelValues, 0, newiPtr, PixelValues.Length);
            bitmap.Save("C:/Depp Data/Others/Wallpaper/Lord of the Ring/" + name + ".bmp", ImageFormat.Bmp);
        }
        // Visualize an image from a GreyScale pixel matrix
        static void VisualizeGreyImage(int[,] list, string name)
        {
            int[,] B = list;
            int[,] G = list;
            int[,] R = list;
            byte[] PixelValues = new byte[iWidth * iHeight * 3];

            int iPoint = 0;

            for (int i = 0; i < iHeight; i++)
            {
                for (int j = 0; j < iWidth; j++)
                {
                    PixelValues[iPoint++] = Convert.ToByte(B[i, j]);
                    PixelValues[iPoint++] = Convert.ToByte(G[i, j]);
                    PixelValues[iPoint++] = Convert.ToByte(R[i, j]);
                }
            }

            Bitmap bitmap = new Bitmap(iWidth, iHeight, PixelFormat.Format24bppRgb);
            Rectangle newrect = new Rectangle(0, 0, iWidth, iHeight);
            BitmapData newbmpData = bitmap.LockBits(newrect,
                ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            IntPtr newiPtr = newbmpData.Scan0;
            System.Runtime.InteropServices.Marshal.Copy(PixelValues, 0, newiPtr, PixelValues.Length);
            bitmap.Save("C:/Depp Data/Others/Wallpaper/Lord of the Ring/" + name + ".jpg", ImageFormat.Jpeg);
        }

        // Compute the gradient of the pixel matrix w.r.t x, y
        static Tuple<double[,], double[,]> GetGradient(int[,] img)
        {
            var gradientX = new double[iHeight, iWidth];
            var gradientY = new double[iHeight, iWidth];
            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    gradientX[i, j] = GradientKernel[0] * img[i, ((j - 2) > 0) ? j - 2 : 0] + GradientKernel[1] * img[i, ((j - 1) > 0) ? j - 1 : 0]
                        + GradientKernel[2] * img[i, j] + GradientKernel[3] * img[i, ((j + 1) < iWidth) ? j + 1 : iWidth - 1]
                        + GradientKernel[4] * img[i, ((j + 2) < iWidth) ? j + 2 : iWidth - 1];
                    gradientY[i, j] = GradientKernel[0] * img[((i - 2) > 0) ? i - 2 : 0, j] + GradientKernel[1] * img[((i - 1) > 0) ? i - 1 : 0, j]
                        + GradientKernel[2] * img[i, j] + GradientKernel[3] * img[((i + 1) < iHeight) ? i + 1 : iHeight - 1, j]
                        + GradientKernel[4] * img[((i + 2) < iHeight) ? i + 2 : iHeight - 1, j];
                }
            return Tuple.Create(gradientX, gradientY);

        }
        // Normalize image to [0, 255]
        static int[,] Normalize(double[,] img)
        {
            double[,] temp_img = new double[iHeight, iWidth];
            int[,] normalized_img = new int[iHeight, iWidth];
            double max = img.Cast<double>().Max();
            double min = img.Cast<double>().Min();
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    temp_img[i, j] = Math.Log10((img[i, j] - min) + 1);
                }
            max = temp_img.Cast<double>().Max();
            min = temp_img.Cast<double>().Min();
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    normalized_img[i, j] = Convert.ToInt32((temp_img[i, j] - min) * 255 / ( max - min));
                }
            return normalized_img;
        }
        // Compute raw Harris value of each pixel in an image
        static double[,] GetHarrisValue(Tuple<double[,], double[,]> gradient_matrix)
        {
            double[,] gradientX = gradient_matrix.Item1;
            double[,] gradientY = gradient_matrix.Item2;
            double[,] Ixx = new double[iHeight, iWidth];
            double[,] Ixy = new double[iHeight, iWidth];
            double[,] Iyy = new double[iHeight, iWidth];
            double[,] HarrisValue = new double[iHeight, iWidth];


            for (int i = 1; i < iHeight - 1; i++)
                for (int j = 1; j < iWidth - 1; j++)
                {
                    Ixx[i, j] = gradientX[i - 1, j - 1] * gradientX[i - 1, j - 1] + gradientX[i - 1, j] * gradientX[i - 1, j] +
                        gradientX[i - 1, j + 1] * gradientX[i - 1, j + 1] + gradientX[i, j - 1] * gradientX[i, j - 1] +
                        gradientX[i, j] * gradientX[i, j] + gradientX[i, j + 1] * gradientX[i, j + 1] + gradientX[i + 1, j - 1] * gradientX[i + 1, j - 1] +
                        gradientX[i + 1, j] * gradientX[i + 1, j] + gradientX[i + 1, j + 1] * gradientX[i + 1, j + 1];
                    Ixy[i, j] = gradientX[i - 1, j - 1] * gradientY[i - 1, j - 1] + gradientX[i - 1, j] * gradientY[i - 1, j] +
                        gradientX[i - 1, j + 1] * gradientY[i - 1, j + 1] + gradientX[i, j - 1] * gradientY[i, j - 1] +
                        gradientX[i, j] * gradientY[i, j] + gradientX[i, j + 1] * gradientY[i, j + 1] + gradientX[i + 1, j - 1] * gradientY[i + 1, j - 1] +
                        gradientX[i + 1, j] * gradientY[i + 1, j] + gradientX[i + 1, j + 1] * gradientY[i + 1, j + 1];
                    Iyy[i, j] = gradientY[i - 1, j - 1] * gradientY[i - 1, j - 1] + gradientY[i - 1, j] * gradientY[i - 1, j] +
                        gradientY[i - 1, j + 1] * gradientY[i - 1, j + 1] + gradientY[i, j - 1] * gradientY[i, j - 1] +
                        gradientY[i, j] * gradientY[i, j] + gradientX[i, j + 1] * gradientY[i, j + 1] + gradientY[i + 1, j - 1] * gradientY[i + 1, j - 1] +
                        gradientY[i + 1, j] * gradientY[i + 1, j] + gradientY[i + 1, j + 1] * gradientY[i + 1, j + 1];
                    HarrisValue[i, j] = Ixx[i, j] * Iyy[i, j] - Ixy[i, j] * Ixy[i, j] - 0.05 * ( Ixx[i, j] + Iyy[i, j] );
                }
            return HarrisValue;
        }
        // Get the Harris detector response without NMS
        static int[,] GetHarrisResponse(double[,] img)
        {
            double mean = 0;
            int[,] response = new int[iHeight, iWidth];

            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    mean += img[i, j];
                }
            mean /= (img.GetLength(0) * img.GetLength(1));

            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    response[i, j] = (img[i, j] > 1.5 * mean) ? 200 : 100;
                }
            return response;
        }
        // Get the Harris detector response with ANMS
        static int[,] GetANMSHarrisResponse(double[,] img)
        {
            int[,] anms_harris_value = new int[iHeight, iWidth];

            if (parallel == true)
            {
                var degreeOfParallelism = 2;
                var tasks = new Task[degreeOfParallelism];
                for (int taskNumber = 0; taskNumber < degreeOfParallelism; taskNumber++)
                {
                    // capturing taskNumber in lambda wouldn't work correctly
                    int taskNumberCopy = taskNumber;
                    tasks[taskNumber] = Task.Factory.StartNew(
                        () =>
                        {
                            double temp_loc_max;
                            int temp_radius;
                            for (int i = (0 + taskNumberCopy) * iHeight / 2; i < (1 + taskNumberCopy) * iHeight / 2; i++)
                            {
                                for (int j = 0; j < iWidth; j++)
                                {
                                    temp_radius = 1;
                                    temp_loc_max = GetMaxValueOfLoop(img, i, j, temp_radius++);
                                    while (img[i, j] / Math.Abs(temp_loc_max) > 1.1 && temp_radius < iHeight / 2)
                                    {
                                        temp_loc_max = (GetMaxValueOfLoop(img, i, j, temp_radius) > temp_loc_max) ? GetMaxValueOfLoop(img, i, j, temp_radius) : temp_loc_max;
                                        temp_radius++;
                                    }
                                    anms_harris_value[i, j] = temp_radius;
                                }
                            }
                        });
                }
                Task.WaitAll(tasks);
            }

            if (parallel == false)
            {
                double temp_loc_max;
                int temp_radius;

                for (int i = 0; i < iHeight; i++)
                {
                    for (int j = 0; j < iWidth; j++)
                    {
                        temp_radius = 1;
                        temp_loc_max = GetMaxValueOfLoop(img, i, j, temp_radius++);
                        while (img[i, j] / Math.Abs(temp_loc_max) > 1.1 && temp_radius < iHeight / 2)
                        {
                            temp_loc_max = (GetMaxValueOfLoop(img, i, j, temp_radius) > temp_loc_max) ? GetMaxValueOfLoop(img, i, j, temp_radius) : temp_loc_max;
                            temp_radius++;
                        }
                        anms_harris_value[i, j] = temp_radius;
                    }
                }
            }
            return anms_harris_value;

        }
        // Get the top K tuple list and pixel matrix with response highlighted
        static Tuple<int[,], List<Tuple<int, int>>> GetTopKValue(int[,] img, int k)
        {
            int[,] result = new int[iHeight, iWidth];
            List<Tuple<int, int>> resultlist = new List<Tuple<int, int>>();
            Tuple<int, int, int>[] tuplearray = new Tuple<int, int, int>[iHeight * iWidth];
            //List<int[,]> test = GetBGRList(image);
            //int[,] result = test[0];

            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    tuplearray[i * iWidth + j] = Tuple.Create(i, j, img[i, j]);
                }

            List<Tuple<int, int, int>> tuplelist = new List<Tuple<int, int, int>>();
            tuplelist.AddRange(tuplearray);

            tuplelist.Sort((x, y) => y.Item3.CompareTo(x.Item3));
            //tuplelist.Reverse();

            for (int i = 0; i < k; i++)
            {
                result[tuplelist[i].Item1, tuplelist[i].Item2] = 255;
                resultlist.Add(Tuple.Create(tuplelist[i].Item1, tuplelist[i].Item2));
            }
            
            return Tuple.Create(result, resultlist);
        }
        // DEPRECATED! Get the top K tuple list
        static List<Tuple<int, int>> GetTopKValueList(double[,] img, int k)
        {
            double[] list = img.Cast<double>().ToArray<double>();
            double[] originallist = img.Cast<double>().ToArray<double>();
            double[] topklist = new double[k];
            List<Tuple<int, int>> result = new List<Tuple<int, int>>();
            //List<int[,]> test = GetBGRList(image);
            //int[,] result = test[0];
            Array.Sort(list);
            Array.Reverse(list);

            Array.Copy(list, topklist, k);
            int num_last_radius = 0;
            int temp_num_last_radius = 0;
            for (int i = 0; i < k; i++)
            {
                if (topklist[k-i-1] == topklist[k-1])
                    num_last_radius++;
                else
                    break;
            }

            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    if (topklist.Contains(originallist[i * img.GetLength(1) + j]))
                    {
                        if (originallist[i * img.GetLength(1) + j] == topklist[k - 1])
                        {
                            temp_num_last_radius++;
                            if (temp_num_last_radius > num_last_radius)
                                continue;
                        }
                        result.Add(Tuple.Create(i, j));
                    }
                }

            return result;
        }
        // Get the max value of a loop
        static double GetMaxValueOfLoop(double[,] img, int x, int y, int r)
        {
            List<double> list = new List<double>();
            Tuple<int, int> temp_loc = new Tuple<int, int>(0, 0);
            for (int j = y - r; j < y + r + 1; j++)
            {
                temp_loc = GetSafeIndex(x - r, j);
                list.Add(img[temp_loc.Item1, temp_loc.Item2]);
                temp_loc = GetSafeIndex(x + r, j);
                list.Add(img[temp_loc.Item1, temp_loc.Item2]);
            }
            for (int i = x - r + 1; i < x + r; i++)
            {
                temp_loc = GetSafeIndex(i, y - r);
                list.Add(img[temp_loc.Item1, temp_loc.Item2]);
                temp_loc = GetSafeIndex(i, y + r);
                list.Add(img[temp_loc.Item1, temp_loc.Item2]);
            }
            return list.Max();
        }
        // Get the safe index of (x, y) to prevent overflow
        static Tuple<int, int> GetSafeIndex(int x, int y)
        {
            int xsafe, ysafe;
            if (x >= 0)
                if (x < iHeight)
                    xsafe = x;
                else
                    xsafe = iHeight - 1;
            else xsafe = 0;
            if (y >= 0)
                if (y < iWidth)
                    ysafe = y;
                else
                    ysafe = iWidth - 1;
            else ysafe = 0;
            return Tuple.Create(xsafe, ysafe);
        }
        // Main function of Harris Keypoint detector with ANMS
        static int[,] ANMSHarrisDetector(Bitmap image, int k)
        {
            List<int[,]> test = GetBGRList(image);
            int[,] anmsharrisvalue = new int[iHeight, iWidth];
            int[,] result = new int[iHeight, iWidth];

            gradientB = GetGradient(test[0]);
            double[,] harrisvalueB = GetHarrisValue(gradientB);
            int [,] anmsB = GetANMSHarrisResponse(harrisvalueB);
            gradientG = GetGradient(test[1]);
            double[,] harrisvalueG = GetHarrisValue(gradientG);
            int[,] anmsG = GetANMSHarrisResponse(harrisvalueG);
            gradientR = GetGradient(test[2]);
            double[,] harrisvalueR = GetHarrisValue(gradientR);
            int[,] anmsR = GetANMSHarrisResponse(harrisvalueR);

            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    anmsharrisvalue[i, j] = anmsB[i, j] + anmsG[i, j] + anmsR[i, j];
                }
            Tuple< int[,], List < Tuple < int, int>>> temptuple = GetTopKValue(anmsharrisvalue, k);
            result = temptuple.Item1;
            topklist = temptuple.Item2;
            //topklist = GetTopKValueList(anmsharrisvalue, k);
            return result;
        }
        // Combine Harris keypoints with original image for visualization
        static List<int[,]> CombineHarrisValueImage(List<int[,]> img, int[,] harrisvalue)
        {
            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    //white points
                    //img[0][i, j] = (img[0][i, j] > harrisvalue[i, j]) ? img[0][i, j] : harrisvalue[i, j];
                    //img[1][i, j] = (img[1][i, j] > harrisvalue[i, j]) ? img[1][i, j] : harrisvalue[i, j];
                    //img[2][i, j] = (img[2][i, j] > harrisvalue[i, j]) ? img[2][i, j] : harrisvalue[i, j];

                    //green points
                    img[0][i, j] = (img[0][i, j] > harrisvalue[i, j]) ? img[0][i, j] : 0;
                    img[1][i, j] = (img[1][i, j] > harrisvalue[i, j]) ? img[1][i, j] : harrisvalue[i, j];
                    img[2][i, j] = (img[2][i, j] > harrisvalue[i, j]) ? img[2][i, j] : 0;
                }
            return img;
        }
        // Compute SIFT feature
        static List<Tuple<int[], int, int>> GetSIFTFeature(List<Tuple<int, int>> harrisvaulelist)
        {
            int subfeature_index;
            List<Tuple<int[], int, int>> feature_list = new List<Tuple<int[], int, int>>();
            foreach (Tuple<int, int> temp_loc in harrisvaulelist)
            {
                int[] temp_sift_feature = new int[384];
                subfeature_index = 0;
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                    {
                        Array.Copy(GetAngleHistogram(gradientB, temp_loc.Item1 + 4 * i - 8, temp_loc.Item2 + 4 * j - 8), 0, temp_sift_feature, 8 * subfeature_index++, 8);
                        Array.Copy(GetAngleHistogram(gradientG, temp_loc.Item1 + 4 * i - 8, temp_loc.Item2 + 4 * j - 8), 0, temp_sift_feature, 8 * subfeature_index++, 8);
                        Array.Copy(GetAngleHistogram(gradientR, temp_loc.Item1 + 4 * i - 8, temp_loc.Item2 + 4 * j - 8), 0, temp_sift_feature, 8 * subfeature_index++, 8);
                    }
                feature_list.Add(Tuple.Create(temp_sift_feature, temp_loc.Item1, temp_loc.Item2));
            }
            return feature_list;
        }
        // Compute the gradient angle histogram of a 4x4 area
        static int[] GetAngleHistogram(Tuple<double[,], double[,]> gradient, int x, int y)
        {
            int[] histogram = new int[8];
            Tuple<int, int> temp_loc = new Tuple<int, int>(0, 0);
            double temp_angle;
            double temp_gradient_x, temp_gradient_y;
            for (int i = x; i < x + 4; i++)
                for (int j = y; j < y + 4; j++)
                {
                    temp_loc = GetSafeIndex(i, j);

                    temp_gradient_x = gradient.Item1[temp_loc.Item1, temp_loc.Item2];
                    temp_gradient_y = gradient.Item2[temp_loc.Item1, temp_loc.Item2];
                    temp_angle = Math.Atan(temp_gradient_y / (temp_gradient_x + 0.01)) - Math.PI / 2;
                    if (temp_gradient_y < 0)
                        temp_angle += Math.PI;
                    if (0 <= temp_angle && temp_angle < Math.PI / 4)
                        histogram[0]++;
                    else if (Math.PI / 4 <= temp_angle && temp_angle < Math.PI / 2)
                        histogram[1]++;
                    else if (Math.PI / 2 <= temp_angle && temp_angle < Math.PI * 3 / 4)
                        histogram[2]++;
                    else if (Math.PI * 3 / 4 <= temp_angle && temp_angle < Math.PI)
                        histogram[3]++;
                    else if (-Math.PI <= temp_angle && temp_angle < -Math.PI * 3 / 4)
                        histogram[4]++;
                    else if (-Math.PI * 3 / 4 <= temp_angle && temp_angle < -Math.PI * 1 / 2)
                        histogram[5]++;
                    else if (-Math.PI * 1 / 2 <= temp_angle && temp_angle < -Math.PI * 1 / 4)
                        histogram[6]++;
                    else if (-Math.PI * 1 / 4 <= temp_angle && temp_angle <= 0)
                        histogram[7]++;
                    else
                    {
                        Console.WriteLine("Angle out of range:" + temp_angle);
                    }
                }
            return histogram;
        }
        // Compute Euclidean Distance
        static int GetEuclideanDistance(int[] feature1, int[] feature2)
        {
            int result = 0;
            for (int i = 0; i < feature1.Length; i++)
                result += (feature1[i] - feature2[i]) * (feature1[i] - feature2[i]);
            return result;
        }
        // Get Top1 match using Euclidean Distance
        static List<Tuple<int, int, int>> GetTop1Match(List<Tuple<int[], int, int>> feature_list1, List<Tuple<int[], int, int>> feature_list2)
        {
            int length1 = feature_list1.Count;
            int length2 = feature_list2.Count;
            int temp_min_distance;
            int temp_best_match;
            int[,] all_distance = new int[length1, length2];
            List<Tuple<int, int, int>> top1match = new List<Tuple<int, int, int>>();
            for (int i = 0; i < length1; i++)
            {
                temp_min_distance = int.MaxValue;
                temp_best_match = 0;
                for (int j = 0; j < length2; j++)
                {
                    all_distance[i, j] = GetEuclideanDistance(feature_list1[i].Item1, feature_list2[j].Item1);
                    if (all_distance[i, j] < temp_min_distance)
                    {
                        temp_best_match = j;
                        temp_min_distance = all_distance[i, j];
                    }
                }
                top1match.Add(Tuple.Create(i, temp_best_match, temp_min_distance));
            }
            return top1match;
        }
        // Visualize match results
        static void VisualizeMatch(List<int[,]> img1, List<int[,]> img2, List<Tuple<int, int, int>> match_list, int k, List<Tuple<int[], int, int>> sift_feature_image_1, List<Tuple<int[], int, int>> sift_feature_image_2)
        {
            int temp_x;
            int temp_y;
            Tuple<int, int> temp_loc = new Tuple<int, int>(0, 0);
            for (int i = 0; i < k; i++)
            {
                List<int[,]> finalmap = new List<int[,]>();
                finalmap.Add(img1[0].Clone() as int[,]);
                finalmap.Add(img1[1].Clone() as int[,]);
                finalmap.Add(img1[2].Clone() as int[,]);
                temp_x = sift_feature_image_1[match_list[i].Item1].Item2;
                temp_y = sift_feature_image_1[match_list[i].Item1].Item3;
                for (int shift_x = -3; shift_x < 4; shift_x++)
                    for (int shift_y = -3; shift_y < 4; shift_y++)
                    {
                        temp_loc = GetSafeIndex(temp_x + shift_x, temp_y + shift_y);
                        finalmap[0][temp_loc.Item1, temp_loc.Item2] = 0;
                        finalmap[1][temp_loc.Item1, temp_loc.Item2] = 255;
                        finalmap[2][temp_loc.Item1, temp_loc.Item2] = 0;

                    }
                VisualizeBGRList(finalmap, "top_match_pair_" + i + "_first_part");

                finalmap.Clear();
                finalmap.Add(img2[0].Clone() as int[,]);
                finalmap.Add(img2[1].Clone() as int[,]);
                finalmap.Add(img2[2].Clone() as int[,]);
                temp_x = sift_feature_image_2[match_list[i].Item2].Item2;
                temp_y = sift_feature_image_2[match_list[i].Item2].Item3;
                for (int shift_x = -3; shift_x < 4; shift_x++)
                    for (int shift_y = -3; shift_y < 4; shift_y++)
                    {
                        temp_loc = GetSafeIndex(temp_x + shift_x, temp_y + shift_y);
                        finalmap[0][temp_loc.Item1, temp_loc.Item2] = 0;
                        finalmap[1][temp_loc.Item1, temp_loc.Item2] = 255;
                        finalmap[2][temp_loc.Item1, temp_loc.Item2] = 0;

                    }
                VisualizeBGRList(finalmap, "top_match_pair_" + i + "_second_part");
            }
        }
    }
}
