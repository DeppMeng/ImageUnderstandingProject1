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
        private class LocationTuple
        {
            private int x, y;
        }


        static void Main(string[] args)
        {
            string str = "C:/Depp Data/Others/Wallpaper/Lord of the Ring/1.jpg";
            GetImage(str);
            //RGB2Grey(image);
            List<int[,]> test = GetBGRList(image);
            //VisualizeBGRList(test, "test1");
            int[,] gradientX = new int[1000, 1600];
            int[,] gradientY = new int[1000, 1600];
            Tuple<int[,], int[,]> gradient = GetGradient(test[0]);
            VisualizeGreyImage(Normalize(gradient.Item1), "gradientx");
            VisualizeGreyImage(Normalize(gradient.Item2), "gradienty");
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
            int iWidth = bitmap.Width;
            int iHeight = bitmap.Height;
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
            int iWidth = bitmap.Width;
            int iHeight = bitmap.Height;
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
            int iHeight = B.GetLength(0);
            int iWidth = B.GetLength(1);
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
            int iHeight = B.GetLength(0);
            int iWidth = B.GetLength(1);
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

        static Tuple<int[,], int[,]> GetGradient(int[,] img)
        {
            int iHeight = img.GetLength(0);
            int iWidth = img.GetLength(1);
            var gradientX = new int[iHeight, iWidth];
            var gradientY = new int[iHeight, iWidth];
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

        static int[,] Normalize(int[,] img)
        {
            double[,] temp_img = new double[img.GetLength(0), img.GetLength(1)];
            int[,] normalized_img = new int[img.GetLength(0), img.GetLength(1)];
            double max = img.Cast<int>().Max();
            double min = img.Cast<int>().Min();
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

        static Tuple<int[,], int[,], int[,]> GetAMatrix(Tuple<int[,], int[,]> gradient_matrix)
        {
            int[,] gradientX = gradient_matrix.Item1;
            int[,] gradientY = gradient_matrix.Item2;
            int iHeight = gradientY.GetLength(0);
            int iWidth = gradientY.GetLength(1);
            int[,] Ixx = new int[iHeight, iWidth];
            int[,] Ixy = new int[iHeight, iWidth];
            int[,] Iyy = new int[iHeight, iWidth];


            for (int i = 0; i < iHeight; i++)
                for (int j = 0; j < iWidth; j++)
                {
                    Ixx[i, j] = gradientX[i, j] * gradientX[i, j];
                    Ixy[i, j] = gradientX[i, j] * gradientY[i, j];
                    Iyy[i, j] = gradientY[i, j] * gradientY[i, j];
                }
            return Tuple.Create(Ixx, Ixy, Iyy);
        }
    }
}
