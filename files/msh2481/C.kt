import org.jetbrains.skija.Canvas
import org.jetbrains.skija.Point
import java.io.File
import kotlin.random.Random

fun interpolate(arr: List<Float>, middlePoints: Int, blurSize: Int) : List<Float> {
    Log("starting", "in interpolate")
    require(middlePoints >= 0) {"middlePoints >= 0"}
    require(blurSize >= 0) {"blurSize >= 0"}
    require(arr.isNotEmpty()) {"arr not empty"}
    var big = MutableList(middlePoints) {arr.first()}
    arr.zipWithNext().forEach{ (l, r) ->
        for (i in 0..middlePoints) {
            big.add(l + (r - l) * i / (middlePoints + 1))
        }
    }
    big.addAll(List(middlePoints + 1) {arr.last()})
    repeat(blurSize) {
        var tmp = big
        for (i in 1..big.size-2) {
            tmp[i] = (big[i - 1] + big[i + 1]) / 2
        }
        big = tmp
    }
    return big
}

fun saveToFile(points: List<Point>, filename: String) {
    Log("starting", "in saveToFile")
    File(filename).writeText(buildString {
        points.forEach { pt ->
            appendLine("${pt.x}, ${pt.y}")
        }
    })
}

fun plotLine(drawer: AxisDrawer, xs: DataSeries, ys: DataSeries, color: Int) {
    Log("starting", "in plotLine")
    val middlePoints = parsedArgs["--middle-points"]?.toIntOrNull() ?: 0
    val blurSize = parsedArgs["--blur-size"]?.toIntOrNull() ?: 0
    val fixedXs = interpolate(xs.data, middlePoints, blurSize)
    val fixedYs = interpolate(ys.data, middlePoints, blurSize)
    val points = (fixedXs zip fixedYs).map{ (x, y) -> Point(x, y) }
    try {
        saveToFile(points, "preprocessed.csv")
    } catch (e: Exception) {
        Log("Can't save preprocessed data", "in plotLine", "error")
        println("Can't save preprocessed data")
    }
    drawer.drawLine(points, color)
    points.forEach{ pt -> drawer.drawPoint(pt.x, pt.y, color) }
}

fun line(canvas: Canvas, canvas2: Canvas, w: Int, h: Int) {
    Log("starting", "in scatter")
    val df = readData(2, 100) ?: return
    val (minX, maxX, minY, maxY) = findXYBounds(df) { it == 0 }
    val drawer = AxisDrawer(canvas, canvas2, w, h, minX, maxX, minY, maxY)

    drawer.drawAxis("x", "y")
    val seededRandom = Random(2)
    val xSeries = df[0]
    val maxColor = 256 * 256 * 256
    val colors = mutableListOf(-maxColor)
    df.drop(1).forEach{ ySeries ->
        val color = seededRandom.nextInt(maxColor) - maxColor
        plotLine(drawer, xSeries, ySeries, color)
        colors.add(color)
    }
    drawer.drawLegend(df.map{ it.name }, colors)
}

import org.jetbrains.skija.Canvas
import kotlin.math.roundToInt

enum class KDEAlgorithm {
    SUM,
    AVERAGE
}

fun blurMatrix(matrix0: Array<FloatArray>, size: Int) : Array<FloatArray> {
    val nCells = matrix0.size
    var matrix = matrix0
    fun getOrZero(x: Int, y: Int) : Float = if ((x in matrix.indices) && (y in matrix[x].indices)) matrix[x][y] else 0f
    repeat(size) {
        val buffer = Array(nCells) { FloatArray(nCells) {0f} }
        for (x in matrix.indices) {
            for (y in matrix[x].indices) {
                val sum = getOrZero(x, y) + getOrZero(x, y-1) + getOrZero(x, y+1) + getOrZero(x-1, y) + getOrZero(x+1, y)
                buffer[x][y] = sum / 5f
            }
        }
        matrix = buffer
    }
    return matrix
}

fun divideMatrices(numerator: Array<FloatArray>, denominator: Array<FloatArray>) : Array<FloatArray> {
    Log("starting", "in divideMatrices")
    for (row in numerator.indices) {
        for (column in numerator[row].indices) {
            if (denominator[row][column] != 0f) {
                numerator[row][column] /= denominator[row][column]
            } else {
                Log("zero denominator encountered", "in divideMatrices", "warn")
                numerator[row][column] = 0f
            }
        }
    }
    return numerator
}

fun kde(canvas: Canvas, canvas2: Canvas, w: Int, h: Int, algo: KDEAlgorithm) {
    Log("starting", "in kde")
    val resolution = parsedArgs["resolution"]?.toIntOrNull() ?: 64
    var df = readData(2, 3) ?: return
    if (df.size == 2) {
        df = df + DataSeries("values", List(df[0].data.size){1f})
    }

    val (minX, maxX, minY, maxY) = findXYBounds(df) { it == 0 }
    fun xToCell(x: Float) : Int = ((x - minX) / (maxX - minX) * (resolution - 1)).roundToInt()
    fun yToCell(y: Float) : Int = ((y - minY) / (maxY - minY) * (resolution - 1)).roundToInt()

    var sumMatrix = Array(resolution) { FloatArray(resolution) { 0f } }
    var cntMatrix = Array(resolution) { FloatArray(resolution) { 0f } }
    (df[0].data zip df[1].data zip df[2].data).forEach { (xy, z) ->
        val x = xToCell(xy.first)
        val y = yToCell(xy.second)
        sumMatrix[x][y] += z
        cntMatrix[x][y] += 1f
    }
    val blurSize = parsedArgs["--blur-size"]?.toIntOrNull() ?: 32
    sumMatrix = blurMatrix(sumMatrix, blurSize)
    if (algo == KDEAlgorithm.AVERAGE) {
        cntMatrix = blurMatrix(cntMatrix, blurSize)
        sumMatrix = divideMatrices(sumMatrix, cntMatrix)
    }

    val drawer = AxisDrawer(canvas, canvas2, w, h, minX, maxX, minY, maxY)
    drawer.drawAxis(df[0].name, df[1].name)
    drawer.drawMatrix(sumMatrix)
}

import org.jetbrains.skija.Canvas
import kotlin.random.Random

/**
 *  Just plots given series of points at the right scale.
 *  Supports multiple series, they will be plotted with different colors
 */
fun scatter(canvas: Canvas, canvas2: Canvas, w: Int, h: Int) {
    Log("starting", "in scatter")
    val df = readData(2, 100) ?: return
    val n = df.size
    if (n % 2 == 1) {
        Log("Expected even number of data series for scatter plot but got $n", "in scatter", "error")
        println("Expected even number of data series for scatter plot but got $n")
        return
    }
    val (minX, maxX, minY, maxY) = findXYBounds(df) { it % 2 == 0 }
    val drawer = AxisDrawer(canvas, canvas2, w, h, minX, maxX, minY, maxY)

    drawer.drawAxis("x", "y")
    val seededRandom = Random(2)
    val colors = mutableListOf<Int>()
    val names = mutableListOf<String>()
    df.chunked(2).forEach{ (xSeries, ySeries) ->
        names.add("${xSeries.name}, ${ySeries.name}")
        if (xSeries.data.size != ySeries.data.size) {
            Log("One of given series have different length for x and y, skipping", "in scatter", "warn")
            println("One of given series have different length for x and y, skipping")
        } else {
            val maxColor = 256 * 256 * 256
            val curColor = seededRandom.nextInt(maxColor) - maxColor
            (xSeries.data zip ySeries.data).forEach { (x, y) ->
                drawer.drawPoint(x, y, curColor)
            }
            colors.add(curColor)
        }
    }
    drawer.drawLegend(names, colors)
}

import kotlinx.datetime.Clock
import org.jetbrains.skija.*
import java.io.File
import java.util.*
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt

/**
 * Date and time up to seconds in format yyyy-mm-ddThh-mm-ss
 */
fun prettyTime() = Clock.System.now().toString().substring(0, 19).replace(':', '-')

/**
 * Minimalistic logger
 * Uses file log{current time}.txt for logging
 * Supports any tags and any predicate determining what should be printed based on these tags
 */
object Log {
    private val file = File("log${prettyTime()}.txt")
    private val tagsCounter: MutableMap<SortedSet<String>, Int> = mutableMapOf()
    var predicate : (Array<out String>) -> Boolean = fun(tags: Array<out String>): Boolean {
        val tagSet = tags.toSortedSet()
        val prevCnt = tagsCounter.getOrDefault(tagSet, 0)
        tagsCounter[tagSet] = prevCnt + 1
        if (prevCnt > 10) {
            return false
        }
        return true
    }
    operator fun invoke(message: String, vararg tags: String) {
        if (predicate(tags)) {
            file.appendText("${prettyTime()} | ${tags.toList()} | $message\n")
        }
    }
}

/**
 * Given interval [l; r] find sequence x_i such that l <= x_i <= r and all x_i are easy to read
 * More precisely, it firstly finds k = 10^t for which (r - l) / k lies in [1; 10] and then divides k by 2 until it lies inn [6; 10]
 * And now ticks are all multiples of k in given range, so we have from 6 to 10 ticks.
 */
fun getTicks(l: Float, r: Float) : List<Float> {
    fun countTicks(k: Float) = (floor(r / k) - ceil(l / k) + 1).roundToInt()
    var k = 1.0f
    while (countTicks(k) > 10) {
        k *= 10
    }
    while (countTicks(k / 10) <= 10) {
        k /= 10
    }
    while (countTicks(k / 2) <= 10) {
        k /= 2
    }
    val result = mutableListOf<Float>()
    for (i in ceil(l / k).toInt()..floor(r / k).toInt()) {
        result.add(i.toFloat() * k)
    }
    return result
}

/**
 * Functional wrapper for colormap: takes number in 0..255, checks this constraint and returns corresponding color from colormap
 */
fun valueToColor(value: Int) : Int {
    check(value in 0..255) { "expected [0;256] but got $value" }
    return DAWN_COLORMAP[value]
}

/**
 * Stores columns of data from input csv file
 */
data class DataSeries(val name: String, val data: List<Float>)

/**
 * Reads comma-separated CSV file and converts it to a list of DataSeries
 */
fun readCSV(filename: String): List<DataSeries>? {
    Log("starting with filename=$filename", "in readCSV")
    val matrix: List<List<String>> = File(filename).readLines().map{ it.split(',') }
    val n = matrix.size
    if (n == 0) {
        Log("The file is empty", "in readCSV", "error")
        println("The file is empty")
        return null
    }
    val m = matrix[0].size
    matrix.forEach {
        if (it.size != m) {
            Log("First row has $m columns, but now found row with ${it.size} columns", "in readCSV", "error")
            println("First row has $m columns, but now found row with ${it.size} columns")
            return null
        }
    }
    Log("read matrix $n x $m", "in readCSV")
    val dataframe = mutableListOf<DataSeries>()
    for (j in 0 until m) {
        val data = mutableListOf<Float>()
        for (i in 1 until n) {
            val cur = matrix[i][j].toFloatOrNull()
            if (cur == null) {
                Log("$i-th element of $j-th data series can't be converted to float: ${matrix[i][j]}", "in readCSV", "warn")
                break
            }
            data.add(cur)
        }
        dataframe.add(DataSeries(matrix[0][j], data))
    }
    return dataframe
}

/**
 * Wrapper for readCSV with more checks:
 * 1. readCSV successfully read file
 * 2. Number of data series lies in given range (minN..maxN)
 * 3. All series are not empty
 */
fun readData(minN: Int, maxN: Int) : List<DataSeries>? {
    Log("starting", "in readData")
    require(minN <= maxN) {"minN <= maxN"}
    val filename = requireNotNull(parsedArgs["--data"]){"--data != null since parseArgs"}
    val df = readCSV(filename)
    if (df == null) {
        Log("Something went wrong when reading csv", "error", "in readData")
        println("Something went wrong when reading csv")
        return null
    }
    if (df.size !in minN..maxN) {
        Log("Need $minN..$maxN data series for this plot but got ${df.size}", "in readData", "error")
        println("Need 2 or 3 data series for kde plot but got ${df.size}")
        return null
    }
    if (df.any{ it.data.isEmpty() }) {
        Log("Need at least one point but got empty series", "in readData", "error")
        println("Need at least one point but got empty series")
        return null
    }
    return df
}

/**
 * First, it finds min and max for a given non-empty sequence, then if they are equal, moves them apart by 1
 */
fun findExtrema(sequence: List<Float>) : List<Float> {
    require(sequence.isNotEmpty()) {"can't find extrema for empty sequence"}
    var min : Float? = null
    var max : Float? = null
    for (elem in sequence) {
        if (min == null || min > elem) {
            min = elem
        }
        if (max == null || max < elem) {
            max = elem
        }
    }
    requireNotNull(min) {"min != null"}
    requireNotNull(max) {"max != null"}
    if (min == max) {
        min -= 1
        max += 1
    }
    return listOf(min, max)
}

/**
 * Applies findExtrema to all series with appropriate indices. "Appropriate" is determined by a given predicate.
 */
fun findXYBounds(df: List<DataSeries>, isXSeries: (Int) -> Boolean) : List<Float> =
    findExtrema(df.filterIndexed{ idx, _ -> isXSeries(idx)}.map{ it.data }.flatten()) +
    findExtrema(df.filterIndexed{ idx, _ -> !isXSeries(idx)}.map{ it.data }.flatten())

/**
 * Applies findExtrema to a given matrix
 */
fun findZBounds(matrix: Array<FloatArray>) : List<Float> = findExtrema(matrix.map{ it.toList() }.flatten())

/**
 * Handles a pair of canvases (for screen and for file, usually) and plenty of constans and functions about them
 */
class AxisDrawer(
    private val canvas: Canvas,
    private val canvas2: Canvas,
    private val w: Int,
    private val h: Int,
    private val minX: Float,
    private val maxX: Float,
    private val minY: Float,
    private val maxY: Float) {

    private val displayMinX = 0.1f * w
    private val displayMaxX = 0.5f * w
    private val displayMinY = 0.1f * h
    private val displayMaxY = 0.9f * h
    private val fontSize = 0.007f * (w + h)
    private val tickW = 0.002f * (w + h)
    private val xTickCaptionOffset = 0.03f * h
    private val yTickCaptionOffset = 0.04f * w
    private val xAxisCaptionX = (displayMinX + displayMaxX) / 2
    private val xAxisCaptionY = displayMaxY + 2 * xTickCaptionOffset
    private val yAxisCaptionX = displayMinX - 2 * yTickCaptionOffset
    private val yAxisCaptionY = (displayMinY + displayMaxY) / 2
    private val legendX = displayMaxX + w * 0.1f
    private val legendY = displayMinY + h * 0.1f
    private val legendOffset = fontSize * 2
    private val legendFrame = 0.02f * (h + w)

    /**
     * Maps data coordinates to screen coordinates
     */
    private fun transformX(x: Float) : Float = displayMinX + (x - minX) / (maxX - minX) * (displayMaxX - displayMinX)
    private fun transformY(y: Float) : Float = displayMaxY + (y - minY) / (maxY - minY) * (displayMinY - displayMaxY)

    private fun drawThinLine(x0: Float, y0: Float, x1: Float, y1: Float) {
        val linePaint = Paint().setARGB(255, 0, 0, 0).setStrokeWidth(1f)
        canvas.drawLine(x0, y0, x1, y1, linePaint)
        canvas2.drawLine(x0, y0, x1, y1, linePaint)
    }
    private fun drawSmallText(x: Float, y: Float, s: String, color: Int = 0xFF000000.toInt()) {
//        val typeface = Typeface.makeFromFile("fonts/JetBrainsMono-Regular.ttf")
        val typeface = FontMgr.getDefault().matchFamilyStyle("Comic Sans MS", FontStyle.NORMAL);
        val font = Font(typeface, fontSize)
        val textPaint = Paint().setColor(color)
        canvas.drawString(s, x, y, font, textPaint)
        canvas2.drawString(s, x, y, font, textPaint)
    }

    fun drawAxis(xName: String, yName: String) {
        Log("starting", "in drawAxis")
        drawThinLine(displayMinX, displayMaxY, displayMaxX, displayMaxY)
        drawThinLine(displayMinX, displayMaxY, displayMinX, displayMinY)
        getTicks(minX, maxX).forEach{ tickX ->
            drawThinLine(transformX(tickX), displayMaxY + tickW, transformX(tickX), displayMaxY)
            drawSmallText(transformX(tickX), displayMaxY + xTickCaptionOffset, "%.2f".format(tickX))
        }
        drawSmallText(xAxisCaptionX, xAxisCaptionY, xName)
        getTicks(minY, maxY).forEach { tickY ->
            drawThinLine(displayMinX, transformY(tickY), displayMinX - tickW, transformY(tickY))
            drawSmallText(displayMinX - yTickCaptionOffset, transformY(tickY), "%.2f".format(tickY))
        }
        drawSmallText(yAxisCaptionX, yAxisCaptionY, yName)
    }

    fun drawLegend(names: List<String>, colors: List<Int>) {
        require(names.size == colors.size) {"|names| = |colors"}
        drawSmallText(legendX, legendY + legendOffset * 0, "min x = $minX")
        drawSmallText(legendX, legendY + legendOffset * 1, "max x = $maxX")
        drawSmallText(legendX, legendY + legendOffset * 2, "min y = $minY")
        drawSmallText(legendX, legendY + legendOffset * 3, "max y = $maxY")
        for (i in names.indices) {
            drawSmallText(legendX, legendY + legendOffset * (i + 5), names[i], colors[i])
        }
        val args = parsedArgs.toList()
        for (i in args.indices) {
            drawSmallText(legendX, legendY + legendOffset * (i + 6 + names.size), "${args[i].first}=${args[i].second}")
        }
        val lx = legendX - legendFrame
        val rx = w - legendFrame
        val ly = legendY - legendFrame
        val ry = legendY + legendOffset * (args.size + names.size + 6) + legendFrame
        drawThinLine(lx, ly, lx, ry)
        drawThinLine(rx, ly, rx, ry)
        drawThinLine(lx, ly, rx, ly)
        drawThinLine(lx, ry, rx, ry)
    }

    fun drawMatrix(matrix: Array<FloatArray>) {
        Log("starting", "in drawMatrix")
        val resolution = matrix.size
        fun cellX(x: Int) : Float = displayMinX + x / resolution.toFloat() * (displayMaxX - displayMinX)
        fun cellY(y: Int) : Float = displayMaxY + y / resolution.toFloat() * (displayMinY - displayMaxY)
        fun drawCell(x: Int, y: Int, sz: Int, value: Int) {
            Log("starting", "in drawKDECell")
            require(value in 0..255) {"0 <= value < 256"}
            val cellColor = valueToColor(value)
            val cellPaint = Paint().setColor(cellColor)
            canvas.drawRect(Rect(cellX(x), cellY(y), cellX(x + sz), cellY(y + sz)), cellPaint)
            canvas2.drawRect(Rect(cellX(x), cellY(y), cellX(x + sz), cellY(y + sz)), cellPaint)
        }
        val (minZ, maxZ) = findZBounds(matrix)
        fun zToValue(z: Float) : Int = ((z - minZ) / (maxZ - minZ) * 255).roundToInt()
        drawCell(0, 0, resolution, 0)
        for (x in 0 until resolution) {
            for (y in 0 until resolution) {
                drawCell(x, y, 1, zToValue(matrix[x][y]))
            }
        }
        drawLegend(listOf("min z = $minZ", "max z = $maxZ"), listOf(0xFF000000.toInt(), 0xFF000000.toInt()))
    }

    fun drawPoint(x: Float, y: Float, color: Int, r: Float = 2f) {
        val pointPaint = Paint().setColor(color)
        canvas.drawCircle(transformX(x), transformY(y), r, pointPaint)
        canvas2.drawCircle(transformX(x), transformY(y), r, pointPaint)
    }

    fun drawLine(p: List<Point>, color: Int, r: Float = 1f) {
        val coords = p.map { pt ->
            val t = Point(transformX(pt.x), transformY(pt.y))
            listOf(t, t)
        }.flatten().drop(1).dropLast(1).toTypedArray()
        val linePaint = Paint().setColor(color).setStrokeWidth(r)
        canvas.drawLines(coords, linePaint)
        canvas2.drawLines(coords, linePaint)
    }
}




import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.swing.Swing
import org.jetbrains.skija.Canvas
import org.jetbrains.skija.EncodedImageFormat
import org.jetbrains.skija.Surface
import org.jetbrains.skiko.SkiaLayer
import org.jetbrains.skiko.SkiaRenderer
import org.jetbrains.skiko.SkiaWindow
import java.awt.Dimension
import java.awt.event.MouseEvent
import java.awt.event.MouseMotionAdapter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.ByteChannel
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import javax.swing.WindowConstants
import kotlin.io.path.Path
import kotlin.system.exitProcess


/** TODO
 * unit test interpolate, getTicks
 * test every parameter
 */

val help = """
    Usage: plot [OPTIONS]
    
    Mandatory options:
    --type=ENUM             ENUM is one of 'scatter', 'line', 'kde-sum', 'kde-average'
    --data=FILE             read input from FILE
    
    Other options:
    --output=FILE           save plot as picture to FILE, no output file by default
    --blur-size=NUM         more blurring for bigger NUM (only for 'line', 'kde-sum', 'kde-average'), 0 by default
    --middle-points=NUM     number of points to add in interpolation (only for 'line'), 0 by default
    --resolution=NUM        use NUMxNUM matrix for KDE (only for 'kde-sum' and 'kde-average'), 64 by default
    --width=NUM             set width of both screen and file pictures to NUM, 1600 by default
    --height=NUM            set height of both screen and file pictures to NUM, width / 2 by default
""".trimIndent()

/**
 * Creates a map from given command line arguments. Expects them in form --name=value.
 */
fun parseArgs(args: Array<String>) : Map<String, String> {
    Log("starting", "in parseArgs")
    val argsMap = mutableMapOf<String, String>()
    for (argument in args) {
        if (argument.count{ it == '='} != 1) {
            Log("Can't understand argument (there should be exactly one '='): $argument", "in parseArgs", "warn")
            println("Can't understand argument (there should be exactly one '='): $argument")
            continue
        }
        val (key, value) = argument.split('=')
        argsMap[key] = value
        Log("$key = $value", "in parseArgs")
    }
    for (mandatory in listOf("--type", "--data")) {
        if (argsMap[mandatory] == null) {
            Log("Missed mandatory option $mandatory, terminating", "in parseArgs", "error")
            println("Missed mandatory option $mandatory, terminating")
            exitProcess(0)
        }
    }
    Log("finishing", "in parseArgs")
    return argsMap
}

var parsedArgs : Map<String, String> = mutableMapOf()
var W = 1600
var H = W / 2

fun main(args: Array<String>) {
    println(KotlinVersion.CURRENT)
    println("v" + System.getProperty("java.version"))
    Log("starting", "in main")
    println(help)
    parsedArgs = parseArgs(args)
    W = parsedArgs["--width"]?.toIntOrNull() ?: 1600
    H = parsedArgs["--height"]?.toIntOrNull() ?: (W / 2)
    createWindow("Your plot")
    Log("finishing", "in main")
}

fun createWindow(title: String) = runBlocking(Dispatchers.Swing) {
    val window = SkiaWindow()
    window.defaultCloseOperation = WindowConstants.DISPOSE_ON_CLOSE
    window.title = title

    window.layer.renderer = Renderer(window.layer)
    window.layer.addMouseMotionListener(MyMouseMotionAdapter)

    window.preferredSize = Dimension(W, H)
    window.minimumSize = Dimension(W,H)
    window.pack()
    window.layer.awaitRedraw()
    window.isVisible = true
}

/**
 * Selects appropriate plotting function from a map based on command line arguments. Also handles all other graphics.
 */
class Renderer(val layer: SkiaLayer): SkiaRenderer {
    override fun onRender(canvas: Canvas, width: Int, height: Int, nanoTime: Long) {
        Log("starting", "in onRender")
        val contentScale = layer.contentScale
        canvas.scale(contentScale, contentScale)
        val w = (width / contentScale).toInt()
        val h = (height / contentScale).toInt()

        val plots = mutableMapOf<String, () -> Unit >()
        val surface = Surface.makeRasterN32Premul(W, H)
        plots["scatter"] = { scatter(canvas, surface.canvas, w, h) }
        plots["kde-sum"] = { kde(canvas, surface.canvas, w, h, KDEAlgorithm.SUM) }
        plots["kde-average"] = { kde(canvas, surface.canvas, w, h, KDEAlgorithm.AVERAGE) }
        plots["line"] = { line(canvas, surface.canvas, w, h) }

        val plotType = requireNotNull(parsedArgs["--type"]) {"--type should be not null since parseArgs"}
        val plotFunc = plots[plotType]
        if (plotFunc == null) {
            Log("No such plot type ($plotType), terminating", "in onRender", "error")
            println("No such plot type ($plotType), terminating")
            exitProcess(0)
        }
        Log("calling plotFunc", "in onRender")
        plotFunc()
        layer.needRedraw()
        Log("writing output file", "in onRender")
        val image = surface.makeImageSnapshot()
        val pngData = image.encodeToData(EncodedImageFormat.PNG)
        val pngBytes: ByteBuffer = pngData!!.toByteBuffer()
        try {
            parsedArgs["--output"]?.let{ output ->
                val path = Path(output)
                val channel: ByteChannel = Files.newByteChannel(
                    path,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE
                )
                channel.write(pngBytes)
                channel.close()
            }
        } catch (e: IOException) {
            println("Failed to write output file")
            Log("caught $e", "in onRender", "error", "exception")
            exitProcess(0)
        }
        Log("starting", "in onRender")
    }
}

object State {
    var mouseX = 0f
    var mouseY = 0f
}

object MyMouseMotionAdapter : MouseMotionAdapter() {
    override fun mouseMoved(event: MouseEvent) {
        State.mouseX = event.x.toFloat()
        State.mouseY = event.y.toFloat()
    }
}