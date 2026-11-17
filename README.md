# Content-Aware Scaling

This project is a small, from-scratch implementation of seam carving, the technique that lets you shrink an image without simply squashing it or cropping off its edges. Instead of scaling every pixel equally, seam carving looks for the parts of the picture that nobody will miss, such as stretches of clear sky, rippling water, or smooth gradients, and quietly removes them one thin line at a time. The interesting content stays intact while the empty space around it collapses.

| Original (4032 × 3024) | Carved (3421 × 1585) |
| :---: | :---: |
| ![Original image](IMG_6081.png) | ![Carved image](carved.png) |
| Taken on iPhone X 2023/09/08 12:01:48 at 50°05'04.4"N 14°24'48.2"E (accurate within 100m) | | 

## Where did the pixels go?

Looking at the two images above, it is fair to wonder whether the carver did anything cleverer than crop the frame. The clearest way to check is to look directly at what was taken out. As the carver works, it records every path it removes, and once it finishes those paths are mapped back onto the original photograph and painted flat grey. Thus the carver functions as a "compressor". It removes pixels in a pathwise fashion and shrinks the image down.

![Removed pixels greyed out on the original image](removed.png)

In this run the carver removed 611 vertical paths and 1,439 horizontal ones, which together account for a little over half of the original pixels, and they are spread across the whole frame rather than taken from its edges. Most of the sky is gone, and the river is covered in a fine net of removed paths, because joins between ripples look just like more ripples. The paths flow around the floating log rather than through it, and they leave the bridge, the castle, and the reflection of the building on the right almost untouched, slipping past the city only through the trees and the willow where a gap costs little. Running `uv run python parse_image.py` writes both `carved.png` and `removed.png`.

## How it works

The heart of the method is a question about each possible cut: if we removed this pixel, would anyone notice? When a pixel is taken out of a row, its left and right neighbours end up side by side, and if the path stepped diagonally to get there, the pixel above also becomes a new neighbour of one of them. Rather than asking how busy the removed pixel was, the carver looks at these new joins and asks how different the pixels on either side of them are. This approach is known as forward energy, because it measures the damage a removal will cause rather than the content it takes away.

A colour difference on its own is not enough to judge visibility, though, because the same step that is glaring in a smooth sky is invisible inside rippling water. So each join is measured against how much its surroundings normally vary. That variability is estimated with the median absolute deviation of neighbouring brightness differences over a window roughly sixty pixels across. Using differences rather than raw brightness means a smooth gradient counts as calm, and using a median means a single sharp edge, such as the floating log, is treated as an outlier instead of inflating the estimate.

Dividing a join's colour difference by the local variability gives a number, z, that says how unusual the join is for its neighbourhood. A join of about one unit is exactly what the region already looks like, so the carver only charges for the part beyond that, using a cost of max(0, z − 1). Joins inside water or sky therefore cost nothing at all, while joins that would break the log, the bridge, or the edge of a building remain expensive. A small floor of one 8-bit brightness level keeps perfectly flat regions from dividing by zero.

With these costs, the next step is to find a seam: a connected path running from the top of the image to the bottom that passes through exactly one pixel per row, where each step may move at most one column to the left or right. Among all such paths, we want the one with the lowest total cost, which makes this a classic dynamic programming problem. Each cell's cost is the cheapest of the three ways of arriving from the row above, each with its own join cost, and once the bottom row is filled in, the cheapest endpoint is traced back up to recover the full path.

Once the best seam is found, it is removed and every pixel to its right shifts over by one, leaving an image that is a single column narrower. This repeats until the cheapest remaining seam is no longer cheap enough, as described under [Deciding when to stop](#deciding-when-to-stop). Shrinking the height works in exactly the same way. Rather than duplicating all of the logic for horizontal seams, `carve_rows` transposes the image so that rows become columns, runs the ordinary vertical carve, and transposes the result back.

## Keeping it fast

A full-resolution phone photo has over twelve million pixels, and every removed seam requires a fresh search across all of them, so the code is built to keep each search cheap.

The join costs are worked out on the fly inside the seam search itself, straight from the image and the variability map, so there is no separate cost map to keep in step with the image. After each removal, only the image and the variability map need to be shifted. The variability map is computed just once at the start, because removing a thin path barely changes how textured a region is, and it is calculated on a grid that samples every fourth pixel before being scaled back up.

The two hot loops, the seam search and the routine that copies pixels around a removed seam, are compiled to machine code with Numba and cached on disk, so only the very first run pays the compilation cost. The seam search is also compiled with relaxed floating-point rules, which on this photo brings a full search down to roughly six hundredths of a second.

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) to manage its dependencies, which are NumPy, Pillow, Numba, and tqdm. It requires Python 3.10 or newer. To install everything, run:

```bash
uv sync
```

The carving script currently reads `IMG_6081.png` from the project directory, so place the image you want to process there under that name (or edit the filename at the bottom of `parse_image.py`). Then run:

```bash
uv run python parse_image.py
```

The script first removes vertical seams to narrow the image, then removes horizontal seams to shorten it, displaying a progress bar along the way that shows the current size and the cost of the most recent seam. When it finishes, it prints how many seams were taken out in each direction, saves the result as `carved.png`, and saves the greyed-out map of removed pixels as `removed.png`.

## Deciding when to stop

Every pass removes the cheapest seam it can find, so seam costs start low and creep upward as the easy material runs out. The question is where to stop, and both `carve` and `carve_rows` take a `stop` argument that answers it in one of three ways.

By default, `stop="free"`, the carver keeps going only while seams are effectively free. If a join sits inside ordinary texture, its z behaves roughly like the size of a draw from a standard normal distribution, and the expected cost of such a join under max(0, z − 1) works out to about 0.167. A seam whose average cost per row stays at or below that leaves joins no more unusual than the texture already around them, so the carver removes it, and it stops at the first seam that costs more. Nothing about this number is tuned; it follows directly from how the cost is defined.

With `stop="elbow"`, the carver instead lets the image's own cost curve choose. It deliberately carves past the useful point, down to a quarter of the original size by default, and records the cost of every seam along the way. The cumulative cost rises slowly while the carver eats filler and steeply once only content is left, so it picks the point that sits furthest below the straight line joining the first and last points, and then rewinds to it by replaying only the seams it kept. This needs no threshold either, but it spends time on seams it later throws away. On the example photo the two methods agree closely: the elbow for the columns falls at seam 1,706, while the free rule stops at 1,726.

Finally, passing a number, as in `stop=0.075`, stops as soon as the cheapest seam's average cost exceeds it. The script currently uses `0.075`, which produced the images above. A stricter limit is sometimes worth it because of one weakness the averages share. A vertical seam is about three thousand rows long and costs nothing through most of them, so its average can stay low even when it spends all of its cost in the few hundred rows where it crosses the city. On this photo the free rule removes about 77% of the pixels and lets a few seams cut visibly through the trees and the bridge, while `0.075` stays clear of them. Judging each seam by its most expensive stretch rather than its overall average would close that loophole. Setting `stop=0` is the strictest option of all, removing only seams in which every single join looks natural.

## Tuning the result

The variability measurement has a few knobs of its own. `mad_radius` and `mad_stride` together set the size of the window used to measure variability, which should be comfortably wider than the thickest edge you want protected, and `mad_floor` sets the smallest variability the carver will assume. An optional `step_penalty` adds a fixed charge to every diagonal step, which keeps seams straighter in smooth gradients; it defaults to zero, since forward energy already charges diagonal steps for any misalignment they cause.

If you would rather stop after a fixed number of seams regardless of their cost, both `carve` and `carve_rows` accept a `max_paths` argument. In elbow mode, `min_keep` sets how far the search carves before looking for the elbow, as a fraction of the original size, and only matters if it is so tight that the elbow is never reached. A `verbose` flag turns the progress bar on or off. Each function returns the carved image together with the list of seams it removed and their average costs.

## Project layout

Nearly all of the work happens in `parse_image.py`, which contains the image loading and saving helpers, the variability estimate, the forward-energy seam search, and the carving loop itself. The same file maps the removed paths back onto the original to produce `removed.png`. It also includes `find_min_paths_from_each_start`, which computes the best seam beginning at every column of the top row in a single pass over any cost matrix; it isn't used by the main carving loop, but it can be useful for exploring the structure of an image.
