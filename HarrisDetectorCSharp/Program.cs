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
        static Bitmap image;
        static int[] GradientKernel = { -2, -1, 0, 1, 2 };
        static int iHeight = 0;
        static int iWidth = 0;


        static void Main(string[] args)
        {
            string str = "C:/Depp Data/Others/Wallpaper/Lord of the Ring/1.jpg";
            GetImage(str);
            iHeight = image.Height;
            iWidth = image.Width;
            //RGB2Grey(image);
            int[,] responsemap = ANMSHarrisDetector(image, 500);
            List<int[,]> finalmap = CombineHarrisValueImage(GetBGRList(image), responsemap);
            VisualizeBGRList(finalmap, "nmsrbg");
            //VisualizeGreyImage(GetTopKValue(finalmap, 500), "nms");
            //GetTopKValue(harrisvalue, 10);
        }

        static void GetImage(string img)
        {
            try
            {
                image = new Bitmap(img);
                return;
            }
            catch (FormatException)
            {
                Console.WriteLine("Image '{0} not found.", img);
                return;
            }
        }

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
            bitmap.Save("C:/Depp Data/Others/Wallpaper/Lord of the Ring/" + name + ".jpg", ImageFormat.Jpeg);
        }

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

        //static List<LocationTuple> HarrisCornerDetector(List<int[,]> img)
        //{
        //    int iHeight = img[0].GetLength(0);
        //    int iWidth = img[0].GetLength(1);
        //    var B = img[0];
        //    var G = img[1];
        //    var R = img[2];
        //    var gradientBX = new int[iHeight, iWidth];
        //    var gradientBY = new int[iHeight, iWidth];
        //    var gradientGX = new int[iHeight, iWidth];
        //    var gradientGY = new int[iHeight, iWidth];
        //    var gradientRX = new int[iHeight, iWidth];
        //    var gradientRY = new int[iHeight, iWidth];
        //    List<int[,]> gradient_list = new List<int[,]>();
        //}

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

        static double[,] GetANMSHarrisResponse(double[,] img)
        {
            double[,] anms_harris_value = new double[iHeight, iWidth];
            double temp_loc_max;
            int temp_radius;
            //double[,] anms_harris_value = new double[iHeight, iWidth];
            //double min = img.Cast<double>().Min();
            //for (int i = radius; i < img.GetLength(0) - radius; i++)
            //    for (int j = radius; j < img.GetLength(1) - radius; j++)
            //    {
            //        anms_harris_value[i, j] = img[i, j] - img[i + 1, j - 1];
            //        for (int k = i - radius; k < i + radius; k++)
            //            for (int m = j - radius; m < j + radius; m++)
            //                if (k != 0 || m != 0)
            //                    anms_harris_value[i, j] = (anms_harris_value[i, j] > img[i, j] - img[k, m]) ? img[i, j] - img[k, m] : anms_harris_value[i, j];
            //        anms_harris_value[i, j] = (anms_harris_value[i, j] / Math.Abs(img[i, j]) > 0) ? img[i, j] : 0;
            //        if (anms_harris_value[i, j] != 0)
            //            Console.WriteLine("here");

            //    }
            for (int i = 0; i < iHeight; i++)
            {
                if (i % 50 == 0)
                    i = i;
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
            return anms_harris_value;

        }

        static int[,] GetTopKValue(double[,] img, int k)
        {
            double[] list = img.Cast<double>().ToArray<double>();
            double[] originallist = img.Cast<double>().ToArray<double>();
            double[] topklist = new double[k];
            int[,] result = new int[iHeight, iWidth];
            //List<int[,]> test = GetBGRList(image);
            //int[,] result = test[0];
            Array.Sort(list);
            Array.Reverse(list);

            Array.Copy(list, topklist, k);

            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    if (topklist.Contains(originallist[i * img.GetLength(1) + j]))
                    {
                        result[i, j] = 255;
                    }
                }

            return result;
        }
        
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

        static int[,] ANMSHarrisDetector(Bitmap image, int k)
        {
            List<int[,]> test = GetBGRList(image);
            double[,] anmsharrisvalue = new double[iHeight, iWidth];
            int[,] result = new int[iHeight, iWidth];

            Tuple<double[,], double[,]> gradientB = GetGradient(test[0]);
            double[,] harrisvalueB = GetHarrisValue(gradientB);
            harrisvalueB = GetANMSHarrisResponse(harrisvalueB);
            Tuple<double[,], double[,]> gradientG = GetGradient(test[1]);
            double[,] harrisvalueG = GetHarrisValue(gradientG);
            harrisvalueG = GetANMSHarrisResponse(harrisvalueG);
            Tuple<double[,], double[,]> gradientR = GetGradient(test[2]);
            double[,] harrisvalueR = GetHarrisValue(gradientR);
            harrisvalueR = GetANMSHarrisResponse(harrisvalueR);

            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    anmsharrisvalue[i, j] = harrisvalueB[i, j] + harrisvalueG[i, j] + harrisvalueR[i, j];
                }
            result = GetTopKValue(anmsharrisvalue, k);
            return result;
        }

        static List<int[,]> CombineHarrisValueImage(List<int[,]> img, int[,] harrisvalue)
        {
            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    img[0][i, j] = (img[0][i, j] > harrisvalue[i, j]) ? img[0][i, j] : harrisvalue[i, j];
                    img[1][i, j] = (img[1][i, j] > harrisvalue[i, j]) ? img[1][i, j] : harrisvalue[i, j];
                    img[2][i, j] = (img[2][i, j] > harrisvalue[i, j]) ? img[2][i, j] : harrisvalue[i, j];
                }
            return img;
        }
    }
}
