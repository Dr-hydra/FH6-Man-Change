using System;
using System.Buffers.Binary;
using System.IO;
using System.Linq;
using BCnEncoder.Decoder;
using BCnEncoder.Encoder;
using BCnEncoder.Shared;
using SkiaSharp;

if (args.Length == 3 && args[0].Equals("decode", StringComparison.OrdinalIgnoreCase))
{
    DecodeSwatchbin(Path.GetFullPath(args[1]), Path.GetFullPath(args[2]));
    return 0;
}

// The Python front-end decodes PNG/TGA/DDS with Pillow and passes exact,
// unpremultiplied RGBA bytes here.  This avoids Skia's premultiplied-alpha
// conversion for transparent source texels (packed maps commonly use those
// RGB bytes as data).
if (args.Length is 6 or 7 && args[0].Equals("encode-raw", StringComparison.OrdinalIgnoreCase))
{
    string templatePath = Path.GetFullPath(args[1]);
    string rawPath = Path.GetFullPath(args[2]);
    if (!int.TryParse(args[3], out int rawWidth) || !int.TryParse(args[4], out int rawHeight))
        throw new ArgumentException("encode-raw width and height must be integers");
    string rawOutputPath = Path.GetFullPath(args[5]);
    // Five arguments after the command are the normal form; the optional GUID
    // is accepted as the final argument when present.
    string? guid = args.Length == 7 ? args[6] : null;
    EncodeRaw(templatePath, rawPath, rawWidth, rawHeight, rawOutputPath, guid);
    return 0;
}

if (args.Length != 3)
{
    Console.Error.WriteLine("Usage: SwatchBinCli <target.swatchbin> <source.png> <output.swatchbin>");
    Console.Error.WriteLine("       SwatchBinCli decode <input.swatchbin> <output.png>");
    Console.Error.WriteLine("       SwatchBinCli encode-raw <template.swatchbin> <rgba.bin> <width> <height> <output.swatchbin> [guid]");
    return 2;
}

string targetPath = Path.GetFullPath(args[0]);
string sourcePath = Path.GetFullPath(args[1]);
string outputPath = Path.GetFullPath(args[2]);
if (File.Exists(outputPath))
    throw new IOException($"Refusing to overwrite {outputPath}");

byte[] target = File.ReadAllBytes(targetPath);
if (target.Length < 0x84 || target.AsSpan(0, 4).SequenceEqual("burG"u8) is false)
    throw new InvalidDataException("Target is not an FH6 swatchbin");

int headerSize = ReadInt32(target, 0x08);
int totalSize = ReadInt32(target, 0x0C);
int width = ReadInt32(target, 0x4C);
int height = ReadInt32(target, 0x50);
int mipCount = target[0x5A];
int encoding = ReadInt32(target, 0x74);
if (headerSize <= 0 || totalSize != target.Length || width <= 0 || height <= 0 || mipCount <= 0)
    throw new InvalidDataException("Target swatchbin header is inconsistent");

CompressionFormat format = GetCompressionFormat(encoding);

using SKBitmap decoded = SKBitmap.Decode(sourcePath)
    ?? throw new InvalidDataException($"Could not decode {sourcePath}");
using var rgba = new SKBitmap(new SKImageInfo(decoded.Width, decoded.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul));
using (var canvas = new SKCanvas(rgba))
{
    canvas.Clear(SKColors.Transparent);
    canvas.DrawBitmap(decoded, 0, 0);
}
using SKBitmap resized = rgba.Resize(
    new SKSizeI(width, height),
    new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None))
    ?? throw new InvalidOperationException("Image resize failed");

byte[] pixels = CopyPixels(resized, width, height);
var encoder = new BcEncoder
{
    OutputOptions =
    {
        Quality = CompressionQuality.BestQuality,
        Format = format,
        GenerateMipMaps = mipCount > 1,
    },
};
byte[][] mips = encoder.EncodeToRawBytes(pixels.AsSpan(), width, height, PixelFormat.Rgba32);
if (mips.Length != mipCount)
    throw new InvalidDataException($"Generated {mips.Length} mips, target requires {mipCount}");

int dataSize = mips.Sum(mip => mip.Length);
int expectedDataSize = target.Length - headerSize;
if (dataSize != expectedDataSize)
    throw new InvalidDataException($"Generated {dataSize} texture bytes, target requires {expectedDataSize}");

byte[] output = new byte[target.Length];
target.AsSpan(0, headerSize).CopyTo(output);
int offset = headerSize;
foreach (byte[] mip in mips)
{
    mip.CopyTo(output, offset);
    offset += mip.Length;
}
WriteInt32(output, 0x0C, output.Length);
WriteInt32(output, 0x24, dataSize);
WriteInt32(output, 0x28, dataSize);

Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
File.WriteAllBytes(outputPath, output);
Console.WriteLine($"SWATCHBIN={outputPath}");
Console.WriteLine($"SOURCE={decoded.Width}x{decoded.Height}");
Console.WriteLine($"TARGET={width}x{height} {format} mips={mipCount} bytes={output.Length}");
return 0;

static int ReadInt32(byte[] data, int offset) => BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset, 4));

static void WriteInt32(byte[] data, int offset, int value) =>
    BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset, 4), value);

static CompressionFormat GetCompressionFormat(int encoding) => encoding switch
{
    0 => CompressionFormat.Bc1,
    1 => CompressionFormat.Bc2,
    2 => CompressionFormat.Bc3,
    3 => CompressionFormat.Bc4,
    5 => CompressionFormat.Bc5,
    9 or 22 => CompressionFormat.Bc7,
    _ => throw new NotSupportedException($"Texture encoding {encoding} is not supported"),
};

static void EncodeRaw(
    string templatePath,
    string rawPath,
    int width,
    int height,
    string outputPath,
    string? guidText)
{
    if (width <= 0 || height <= 0)
        throw new ArgumentOutOfRangeException(nameof(width), "Raw dimensions must be positive");
    if (File.Exists(outputPath))
        throw new IOException($"Refusing to overwrite {outputPath}");

    byte[] target = File.ReadAllBytes(templatePath);
    if (target.Length < 0x84 || target.AsSpan(0, 4).SequenceEqual("burG"u8) is false)
        throw new InvalidDataException("Target is not an FH6 swatchbin");
    int headerSize = ReadInt32(target, 0x08);
    int targetWidth = ReadInt32(target, 0x4C);
    int targetHeight = ReadInt32(target, 0x50);
    int mipCount = target[0x5A];
    int encoding = ReadInt32(target, 0x74);
    CompressionFormat format = GetCompressionFormat(encoding);
    if (width != targetWidth || height != targetHeight)
        throw new InvalidDataException(
            $"Raw dimensions {width}x{height} do not match template {targetWidth}x{targetHeight}; " +
            "resize in the front-end so the operation is explicit");

    byte[] pixels = File.ReadAllBytes(rawPath);
    long expectedPixels = (long)width * height * 4;
    if (pixels.LongLength != expectedPixels)
        throw new InvalidDataException(
            $"Raw RGBA byte count {pixels.LongLength} does not equal {expectedPixels}");

    var encoder = new BcEncoder
    {
        OutputOptions =
        {
            Quality = CompressionQuality.BestQuality,
            Format = format,
            GenerateMipMaps = mipCount > 1,
        },
    };
    byte[][] mips = encoder.EncodeToRawBytes(pixels.AsSpan(), width, height, PixelFormat.Rgba32);
    if (mips.Length != mipCount)
        throw new InvalidDataException($"Generated {mips.Length} mips, target requires {mipCount}");
    int dataSize = mips.Sum(mip => mip.Length);
    int expectedDataSize = target.Length - headerSize;
    if (dataSize != expectedDataSize)
        throw new InvalidDataException(
            $"Generated {dataSize} texture bytes, target requires {expectedDataSize}");

    byte[] output = new byte[target.Length];
    target.AsSpan(0, headerSize).CopyTo(output);
    int offset = headerSize;
    foreach (byte[] mip in mips)
    {
        mip.CopyTo(output, offset);
        offset += mip.Length;
    }
    WriteInt32(output, 0x0C, output.Length);
    WriteInt32(output, 0x24, dataSize);
    WriteInt32(output, 0x28, dataSize);
    if (guidText is not null)
    {
        if (!Guid.TryParse(guidText, out Guid guid))
            throw new ArgumentException($"Invalid output GUID {guidText}");
        guid.ToByteArray().CopyTo(output, 0x3C);
    }

    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    File.WriteAllBytes(outputPath, output);
    Console.WriteLine($"SWATCHBIN={outputPath}");
    Console.WriteLine($"SOURCE_RAW={width}x{height} bytes={pixels.Length}");
    Console.WriteLine($"TARGET={targetWidth}x{targetHeight} {format} mips={mipCount} bytes={output.Length}");
    if (guidText is not null)
        Console.WriteLine($"GUID={guidText}");
}

static void DecodeSwatchbin(string inputPath, string outputPath)
{
    if (File.Exists(outputPath))
        throw new IOException($"Refusing to overwrite {outputPath}");
    byte[] input = File.ReadAllBytes(inputPath);
    if (input.Length < 0x84 || input.AsSpan(0, 4).SequenceEqual("burG"u8) is false)
        throw new InvalidDataException("Input is not an FH6 swatchbin");
    int headerSize = ReadInt32(input, 0x08);
    int width = ReadInt32(input, 0x4C);
    int height = ReadInt32(input, 0x50);
    CompressionFormat format = GetCompressionFormat(ReadInt32(input, 0x74));
    var decoder = new BcDecoder();
    var decoded = decoder.DecodeRaw(input.AsSpan(headerSize).ToArray(), width, height, format);
    if (decoded.Length != width * height)
        throw new InvalidDataException("Decoded pixel count does not match texture dimensions");

    byte[] pixels = new byte[decoded.Length * 4];
    for (int index = 0; index < decoded.Length; index++)
    {
        pixels[index * 4] = decoded[index].r;
        pixels[index * 4 + 1] = decoded[index].g;
        pixels[index * 4 + 2] = decoded[index].b;
        pixels[index * 4 + 3] = decoded[index].a;
    }
    using var bitmap = new SKBitmap(new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Unpremul));
    System.Runtime.InteropServices.Marshal.Copy(pixels, 0, bitmap.GetPixels(), pixels.Length);
    using SKImage image = SKImage.FromBitmap(bitmap);
    using SKData png = image.Encode(SKEncodedImageFormat.Png, 100);
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    using FileStream output = File.OpenWrite(outputPath);
    png.SaveTo(output);
    Console.WriteLine($"PNG={outputPath}");
    Console.WriteLine($"SOURCE={width}x{height} {format}");
}

static byte[] CopyPixels(SKBitmap bitmap, int width, int height)
{
    int packedRowBytes = width * 4;
    byte[] result = new byte[packedRowBytes * height];
    IntPtr pointer = bitmap.GetPixels();
    if (pointer == IntPtr.Zero)
        throw new InvalidOperationException("Could not access resized pixels");
    if (bitmap.RowBytes == packedRowBytes)
    {
        System.Runtime.InteropServices.Marshal.Copy(pointer, result, 0, result.Length);
        return result;
    }

    byte[] source = new byte[bitmap.RowBytes * height];
    System.Runtime.InteropServices.Marshal.Copy(pointer, source, 0, source.Length);
    for (int row = 0; row < height; row++)
        Buffer.BlockCopy(source, row * bitmap.RowBytes, result, row * packedRowBytes, packedRowBytes);
    return result;
}
