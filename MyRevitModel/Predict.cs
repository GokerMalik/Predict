using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Autodesk.Revit.Attributes;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static System.Net.Mime.MediaTypeNames;
using System.Security.Cryptography;

namespace MyRevitModel
{
    [TransactionAttribute(TransactionMode.ReadOnly)]
    public class Predict : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            //Get UIDocument
            UIDocument uidoc = commandData.Application.ActiveUIDocument;

            //Get Document
            Document doc = uidoc.Document;
            //Pick Object
            Reference pickedObj = uidoc.Selection.PickObject(Autodesk.Revit.UI.Selection.ObjectType.Element);
            double[] lengths = new double[3];
            EdgeArray gEdges = new EdgeArray();

            if (pickedObj != null)
            {
                //Retrieve Element
                ElementId eleId = pickedObj.ElementId;
                Element ele = doc.GetElement(eleId);
                string selCategory = ele.Category.Name;

                //try to get the element and its edges
                if (selCategory == "Mass")
                {
                    try
                    {
                        //get all the edges inside the element
                        gEdges = getEdges(ele);
                        //get lengths
                        lengths = getDimensions(gEdges);
                    }
                    catch (Exception e)
                    {
                        message = e.Message;
                        return Result.Failed;
                    }
                }
                else
                {
                    TaskDialog.Show("Error", "This tool works only with mass category, for now");
                    return Result.Failed;
                }
            }
            else
            {
                TaskDialog.Show("Error", "No Object Selected");
                return Result.Failed;
            }

            string AssPath = this.GetType().Assembly.Location;
            string modelPath = AssPath.Substring(0, AssPath.LastIndexOf(@"\")) + @"\model.onnx";

            try
            {
                (int[] dataInfo, float[] probabilities) = PredictCategory(modelPath, lengths);

                int prediction = probabilities.ToList().IndexOf(probabilities.Max());
                string percentage = (probabilities.Max() * 100).ToString() + "%";

                if (prediction == 0)
                {
                    TaskDialog.Show("Result", "A wall with the chance of " + percentage + Environment.NewLine
                        + Environment.NewLine + "Dim1: " + dataInfo[0].ToString()
                        + Environment.NewLine + "Dim2: " + dataInfo[1].ToString()
                        + Environment.NewLine + "Dim3: " + dataInfo[2].ToString());
                }

                if (prediction == 1)
                {
                    TaskDialog.Show("Result", "A floor with the chance of " + percentage + Environment.NewLine
                        + Environment.NewLine + "Dim1: " + dataInfo[0].ToString()
                        + Environment.NewLine + "Dim2: " + dataInfo[1].ToString()
                        + Environment.NewLine + "Dim3: " + dataInfo[2].ToString());
                }

                if (prediction == 2)
                {
                    TaskDialog.Show("Result", "A column with the chance of " + percentage + Environment.NewLine
                        + Environment.NewLine + "Dim1: " + dataInfo[0].ToString()
                        + Environment.NewLine + "Dim2: " + dataInfo[1].ToString()
                        + Environment.NewLine + "Dim3: " + dataInfo[2].ToString());
                }

                if (prediction == 3)
                {
                    TaskDialog.Show("Result", "A beam with the chance of " + percentage + Environment.NewLine
                        + Environment.NewLine + "Dim1: " + dataInfo[0].ToString()
                        + Environment.NewLine + "Dim2: " + dataInfo[1].ToString()
                        + Environment.NewLine + "Dim3: " + dataInfo[2].ToString());
                }
            }
            catch(Exception e1)
            {
                message = e1.Message;
                return Result.Failed;
            }

            return Result.Succeeded;
        }

        public EdgeArray getEdges(Element ele)
        {
            Options gOptions = new Options();
            gOptions.DetailLevel = ViewDetailLevel.Fine;

            GeometryElement geom = ele.get_Geometry(gOptions);

            //create an Edge array
            EdgeArray gEdges = new EdgeArray();

            //check each geometry object inside the geom to find the one with edges
            foreach (GeometryObject gObj in geom)
            {
                Solid gSolid = gObj as Solid;

                if (gSolid != null)
                {
                    gEdges = (gSolid.Edges.IsEmpty) ? gEdges : gSolid.Edges;
                }
            }

            return gEdges;

        }
        public double[] getDimensions(EdgeArray gEdges)
        {
            double[] CurLengths = new double[3];
            for (int i = 0; i < 12; i++)
            {
                XYZ start = gEdges.get_Item(i).AsCurve().GetEndPoint(0);
                XYZ end = gEdges.get_Item(i).AsCurve().GetEndPoint(1);

                double curLength = gEdges.get_Item(i).AsCurve().Length;
                double implement = UnitUtils.ConvertFromInternalUnits(curLength, UnitTypeId.Millimeters);

                if (Math.Round(start.X * 100) != Math.Round(end.X * 100))
                {
                    CurLengths[0] = implement;
                }
                else if (Math.Round(start.Y * 100) != Math.Round(end.Y * 100))
                {
                    CurLengths[1] = implement;
                }
                else
                {
                    CurLengths[2] = implement;
                }
            }

            return CurLengths;
        }
        private static (int[], float[]) PredictCategory(string modelPath, double[] EdgeDims)
        {
            InferenceSession InferSess = new InferenceSession(modelPath);

            string modelInputLayerName = InferSess.InputMetadata.Keys.Single();
            int[] dimensions = { 1, 3 };

            int[] newDims = new int[3];
            newDims[0] = (int)(EdgeDims[0]);
            newDims[1] = (int)(EdgeDims[1]);
            newDims[2] = (int)(EdgeDims[2]);

            DenseTensor<int> inputTensor = new DenseTensor<int>(newDims, dimensions, false);
            //DenseTensor<float> inputTensor = new DenseTensor<float>(newDims, dimensions, false);

            Memory<int> memoryBlock = inputTensor.Buffer;
            int[] DataInfo = memoryBlock.ToArray();

            List<NamedOnnxValue> modelInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(modelInputLayerName, inputTensor)
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = InferSess.Run(modelInput);

            return (DataInfo,((DenseTensor<float>) result.Single().Value).ToArray());

        }
    }
}