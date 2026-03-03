import copy
import logging
from itertools import groupby

import docling_ibm_models.tableformer.settings as s

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG
logger = s.get_custom_logger("consolidate", LOG_LEVEL)
# png_files = {}  # Evaluation files
total_pics = 0


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def otsl_clean(rs_list):
    new_rs_list = []
    stop_list = ["<pad>", "<unk>", "<start>", "<end>"]
    for tag in rs_list:
        if tag not in stop_list:
            new_rs_list.append(tag)
    return new_rs_list


def otsl_sqr_chk(rs_list, name, logdebug):
    rs_list_split = [
        list(group) for k, group in groupby(rs_list, lambda x: x == "nl") if not k
    ]
    isSquare = True
    if len(rs_list_split) > 0:
        init_tag_len = len(rs_list_split[0]) + 1
        for ind, ln in enumerate(rs_list_split):
            ln.append("nl")
            if len(ln) != init_tag_len:
                isSquare = False
        if isSquare:
            if logdebug:
                logger.debug(
                    "{}*OK* Table is square! *OK*{}".format(
                        bcolors.OKGREEN, bcolors.ENDC
                    )
                )
        else:
            err_name = "{}*ERR* " + name + " *ERR*{}"
            logger.debug(err_name.format(bcolors.FAIL, bcolors.ENDC))
            logger.debug(
                "{}*ERR* Table is not square! *ERR*{}".format(
                    bcolors.FAIL, bcolors.ENDC
                )
            )
    return isSquare


def otsl_pad_to_sqr(rs_list, pad_tag):
    new_list = []
    rs_list_split = [
        list(group) for k, group in groupby(rs_list, lambda x: x == "nl") if not k
    ]
    max_row_len = 0
    for ind, ln in enumerate(rs_list_split):
        if len(ln) > max_row_len:
            max_row_len = len(ln)
    for ind, ln in enumerate(rs_list_split):
        ln += [pad_tag] * (max_row_len - len(ln))
        ln.append("nl")
        new_list.extend(ln)
    return new_list


def otsl_tags_cells_sync_chk(rs_list, cells, name, logdebug):
    countCellTags = 0
    isGood = True
    for rsTag in rs_list:
        if rsTag in ["fcel", "ched", "rhed", "srow", "ecel"]:
            countCellTags += 1
    if countCellTags != len(cells):
        err_name = "{}*!ERR* " + name + " *ERR!*{}"
        logger.debug(err_name.format(bcolors.FAIL, bcolors.ENDC))
        err_msg = "{}*!ERR* Tags are not in sync with cells! *ERR!*{}"
        logger.debug(err_msg.format(bcolors.FAIL, bcolors.ENDC))
        isGood = False
    return isGood


def otsl_check_down(rs_split, x, y):
    distance = 1
    elem = "ucel"
    goodlist = ["fcel", "ched", "rhed", "srow", "ecel", "lcel", "nl"]
    while elem not in goodlist and y < len(rs_split) - 1:
        y += 1
        distance += 1
        elem = rs_split[y][x]
    if elem in goodlist:
        distance -= 1
    return distance


def otsl_check_right(rs_split, x, y):
    distance = 1
    elem = "lcel"
    goodlist = ["fcel", "ched", "rhed", "srow", "ecel", "ucel", "nl"]
    while elem not in goodlist and x < (len(rs_split[y]) - 1):
        x += 1
        distance += 1
        elem = rs_split[y][x]
    if elem in goodlist:
        distance -= 1
    return distance


def otsl_to_html(rs_list, logdebug):
    if len(rs_list) == 0:
        return []

    if rs_list[0] not in ["fcel", "ched", "rhed", "srow", "ecel"]:
        # Most likely already HTML...
        return rs_list
    html_table = []
    if logdebug:
        logger.debug(
            "{}*Reconstructing HTML...*{}".format(bcolors.WARNING, bcolors.ENDC)
        )

    if not otsl_sqr_chk(rs_list, "---", logdebug):
        # PAD TABLE TO SQUARE
        logger.debug("{}*Padding to square...*{}".format(bcolors.WARNING, bcolors.ENDC))
        rs_list = otsl_pad_to_sqr(rs_list, "lcel")

    # 2D structure, line by line:
    rs_list_split = [
        list(group) for k, group in groupby(rs_list, lambda x: x == "nl") if not k
    ]

    if logdebug:
        logger.debug("")

    # Sequentially store indexes of 2D spans that were registered to avoid re-registering them
    registry_2d_span = []

    # Iterate all elements in the rs line, and look right / down to detect spans
    # If span detected - run function to find size of the span
    # repeat with all cells
    thead_present = False

    for rs_row_ind, rs_row in enumerate(rs_list_split):
        html_list = []

        if not thead_present:
            if "ched" in rs_list_split[rs_row_ind]:
                html_list.append("<thead>")
                thead_present = True

        if thead_present:
            if "ched" not in rs_list_split[rs_row_ind]:
                html_list.append("</thead>")
                thead_present = False

        html_list.append("<tr>")
        for rs_cell_ind, rs_cell in enumerate(rs_list_split[rs_row_ind]):
            if rs_cell in ["fcel", "ched", "rhed", "srow", "ecel"]:
                rdist = 0
                ddist = 0
                xrdist = 0
                xddist = 0
                span = False
                # Check if it has horizontal span:
                if rs_cell_ind + 1 < len(rs_list_split[rs_row_ind]):
                    if rs_list_split[rs_row_ind][rs_cell_ind + 1] == "lcel":
                        rdist = otsl_check_right(rs_list_split, rs_cell_ind, rs_row_ind)
                        span = True
                # Check if it has vertical span:
                if rs_row_ind + 1 < len(rs_list_split):
                    # logger.debug(">>>")
                    # logger.debug(rs_list_split[rs_row_ind + 1])
                    # logger.debug(">>> rs_cell_ind = {}".format(rs_cell_ind))
                    if rs_list_split[rs_row_ind + 1][rs_cell_ind] == "ucel":
                        ddist = otsl_check_down(rs_list_split, rs_cell_ind, rs_row_ind)
                        span = True
                # Check if it has 2D span:
                if rs_cell_ind + 1 < len(rs_list_split[rs_row_ind]):
                    if rs_list_split[rs_row_ind][rs_cell_ind + 1] == "xcel":
                        xrdist = otsl_check_right(
                            rs_list_split, rs_cell_ind, rs_row_ind
                        )
                        xddist = otsl_check_down(rs_list_split, rs_cell_ind, rs_row_ind)
                        span = True
                        # Check if this 2D span was already registered,
                        # If not - register, if yes - cancel span
                        # logger.debug("rs_cell_ind: {}, xrdist:{}".format(rs_cell_ind, xrdist))
                        # logger.debug("rs_row_ind: {}, xddist:{}".format(rs_cell_ind, xrdist))
                        for x in range(rs_cell_ind, xrdist + rs_cell_ind):
                            for y in range(rs_row_ind, xddist + rs_row_ind):
                                reg2dind = str(x) + "_" + str(y)
                                # logger.debug(reg2dind)
                                if reg2dind in registry_2d_span:
                                    # Cell of the span is already in, cancel current span
                                    span = False
                        if span:
                            # None of the span cells were previously registered
                            # Register an entire span
                            for x in range(rs_cell_ind, xrdist + rs_cell_ind):
                                for y in range(rs_row_ind, xddist + rs_row_ind):
                                    reg2dind = str(x) + "_" + str(y)
                                    registry_2d_span.append(reg2dind)
                if span:
                    html_list.append("<td")
                    if rdist > 1:
                        html_list.append(' colspan="' + str(rdist) + '"')
                    if ddist > 1:
                        html_list.append(' rowspan="' + str(ddist) + '"')
                    if xrdist > 1:
                        html_list.append(' rowspan="' + str(xddist) + '"')
                        html_list.append(' colspan="' + str(xrdist) + '"')
                    html_list.append(">")
                    html_list.append("</td>")
                else:
                    html_list.append("<td>")
                    html_list.append("</td>")
        html_list.append("</tr>")
        html_table.extend(html_list)

    if logdebug:
        logger.debug(
            "*********************** registry_2d_span ***************************"
        )
        logger.debug(registry_2d_span)
        logger.debug(
            "********************************************************************"
        )

    return html_table


def html_to_otsl(table, writer, logdebug, extra_debug, include_html, use_writer):
    r"""
    Converts table structure from HTML to RS

    Parameters
    ----------
    table : json
        line from jsonl
    writer : writer
        Writes lines into output jsonl
    """

    table_html_structure = copy.deepcopy(table["html"]["structure"])
    out_line = table
    if include_html:
        out_line["html"]["html_structure"] = table_html_structure
        out_line["html"]["html_restored_structure"] = {"tokens": []}

    out_line["html"]["structure"] = {"tokens": []}
    # possible colspans
    pos_colspans = {
        ' colspan="20"': 20,
        ' colspan="19"': 19,
        ' colspan="18"': 18,
        ' colspan="17"': 17,
        ' colspan="16"': 16,
        ' colspan="15"': 15,
        ' colspan="14"': 14,
        ' colspan="13"': 13,
        ' colspan="12"': 12,
        ' colspan="11"': 11,
        ' colspan="10"': 10,
        ' colspan="2"': 2,
        ' colspan="3"': 3,
        ' colspan="4"': 4,
        ' colspan="5"': 5,
        ' colspan="6"': 6,
        ' colspan="7"': 7,
        ' colspan="8"': 8,
        ' colspan="9"': 9,
    }
    # possible rowspans
    pos_rowspans = {
        ' rowspan="20"': 20,
        ' rowspan="19"': 19,
        ' rowspan="18"': 18,
        ' rowspan="17"': 17,
        ' rowspan="16"': 16,
        ' rowspan="15"': 15,
        ' rowspan="14"': 14,
        ' rowspan="13"': 13,
        ' rowspan="12"': 12,
        ' rowspan="11"': 11,
        ' rowspan="10"': 10,
        ' rowspan="2"': 2,
        ' rowspan="3"': 3,
        ' rowspan="4"': 4,
        ' rowspan="5"': 5,
        ' rowspan="6"': 6,
        ' rowspan="7"': 7,
        ' rowspan="8"': 8,
        ' rowspan="9"': 9,
    }

    t_cells = []  # 2D structure
    tl_cells = []  # 1D structure
    t_expands = []  # 2D structure
    tl_spans = {}  # MAP, POPULATE WITH ACTUAL SPANS VALUES, IN SYNC WITH tl_cells

    current_line = 0
    current_column = 0
    current_html_cell_ind = 0

    current_line_tags = []
    current_line_expands = []

    if logdebug:
        logger.debug("")
        logger.debug("*** {}: {} ***".format(table["split"], table["filename"]))

    colnum = 0

    if extra_debug:
        logger.debug(
            "========================== Input HTML ============================"
        )
        logger.debug(table_html_structure["tokens"])
        logger.debug(
            "=================================================================="
        )

    if logdebug:
        logger.debug("********")
        logger.debug("* OTSL *")
        logger.debug("********")

    for i in range(len(table_html_structure["tokens"])):
        html_tag = table_html_structure["tokens"][i]
        prev_html_tag = ""
        next_html_tag = ""
        if i > 0:
            prev_html_tag = table_html_structure["tokens"][i - 1]
        if i < len(table_html_structure["tokens"]) - 1:
            next_html_tag = table_html_structure["tokens"][i + 1]

        if html_tag not in ["<thead>", "<tbody>"]:
            # Then check the next tag...
            # rules of conversion
            # Check up-cell in t_expands, in case row-spans have to be inserted
            if html_tag in ["<td>", "<td", "</tr>"]:
                if current_line > 0:
                    if current_column >= len(t_expands[current_line - 1]):
                        # !!!
                        return False, {}
                    up_expand = t_expands[current_line - 1][current_column]

                    while up_expand[1] > 0:
                        if up_expand[0] == 0:
                            # ucel
                            current_line_tags.append("ucel")
                            current_line_expands.append([0, up_expand[1] - 1])
                            current_column += 1
                        else:
                            # xcel
                            for ci in range(up_expand[0]):
                                current_line_tags.append("xcel")
                                current_line_expands.append(
                                    [up_expand[0] - ci, up_expand[1] - 1]
                                )
                                current_column += 1
                        up_expand = t_expands[current_line - 1][current_column]
            # ======================================================================================
            # Fix for trailing "ucel" in a row
            if html_tag in ["</tr>"]:
                if current_line > 0:
                    cur_line_len = len(current_line_expands)
                    pre_line_len = len(t_expands[current_line - 1])

                    if cur_line_len < pre_line_len:
                        extra_columns = pre_line_len - cur_line_len - 1
                        if extra_columns > 0:
                            if extra_debug:
                                logger.debug(
                                    "Extra columns needed in row: {}".format(
                                        extra_columns
                                    )
                                )

                            for clm in range(extra_columns):
                                up_expand = t_expands[current_line - 1][
                                    cur_line_len + clm
                                ]
                                if up_expand[0] == 0:
                                    # ucel
                                    current_line_tags.append("ucel")
                                    current_line_expands.append([0, up_expand[1] - 1])
                                else:
                                    # xcel
                                    current_line_tags.append("xcel")
                                    current_line_expands.append(
                                        [up_expand[0], up_expand[1] - 1]
                                    )
            # ======================================================================================

            # 1. Opening cell tags
            if html_tag in ["<td>", "<td"]:
                # check if cell is empty...
                cell_is_empty = True
                if "cells" in table["html"]:
                    cell_tokens = table["html"]["cells"][current_html_cell_ind][
                        "tokens"
                    ]
                else:
                    cell_tokens = "f"

                # Clean cell_tokens from trash:
                cell_tokens = list(filter(lambda a: a != "<i>", cell_tokens))
                cell_tokens = list(filter(lambda a: a != "<I>", cell_tokens))
                cell_tokens = list(filter(lambda a: a != "<b>", cell_tokens))
                cell_tokens = list(filter(lambda a: a != "<B>", cell_tokens))
                cell_tokens = list(filter(lambda a: a != " ", cell_tokens))
                cell_tokens = list(filter(lambda a: a != "</b>", cell_tokens))
                cell_tokens = list(filter(lambda a: a != "</B>", cell_tokens))
                cell_tokens = list(filter(lambda a: a != "</i>", cell_tokens))
                cell_tokens = list(filter(lambda a: a != "</I>", cell_tokens))

                # Check if empty
                if len(cell_tokens) > 0:
                    cell_is_empty = False
                if cell_is_empty:
                    out_line["html"]["cells"][current_html_cell_ind]["tokens"] = []
                    current_line_tags.append("ecel")
                    current_line_expands.append([0, 0])
                else:
                    current_line_tags.append("fcel")
                    current_line_expands.append([0, 0])
                current_html_cell_ind += 1
                current_column += 1

            # 2. Closing row tags
            if html_tag == "</tr>":
                if len(current_line_tags) > colnum:
                    colnum = len(current_line_tags)
                # Save everything we read about the line to t_cells
                current_line_tags.append("nl")
                t_cells.append(copy.deepcopy(current_line_tags))
                tl_cells.extend(copy.deepcopy(current_line_tags))
                if logdebug:
                    print(current_line_tags)
                current_line_tags = []

                # Deal with expands
                current_line_expands.append([-1, -1])
                # Output spans metadata
                t_expands.append(copy.deepcopy(current_line_expands))
                current_line_expands = []

                current_column = 0
                current_line += 1
            # 3. Colspans only
            if html_tag in pos_colspans:
                if prev_html_tag not in pos_rowspans:
                    if next_html_tag not in pos_rowspans:
                        colspan_len = pos_colspans[html_tag]
                        tl_spans[current_html_cell_ind - 1] = [colspan_len, 1]
                        current_line_expands[len(current_line_expands) - 1] = [
                            colspan_len,
                            0,
                        ]
                        for ci in range(colspan_len - 1):
                            current_line_tags.append("lcel")
                            current_line_expands.append([colspan_len - ci - 1, 0])
                            current_column += 1

            # 4. Rowspans only
            if html_tag in pos_rowspans:
                if prev_html_tag not in pos_colspans:
                    if next_html_tag not in pos_colspans:
                        rowspan_len = pos_rowspans[html_tag]
                        tl_spans[current_html_cell_ind - 1] = [1, rowspan_len]
                        current_line_expands[len(current_line_expands) - 1] = [
                            0,
                            rowspan_len - 1,
                        ]

            # 5. 2D spans
            if html_tag in pos_rowspans:
                rowspan_len = pos_rowspans[html_tag]
                if prev_html_tag in pos_colspans:
                    colspan_len = pos_colspans[prev_html_tag]
                    tl_spans[current_html_cell_ind - 1] = [colspan_len, rowspan_len]
                    newexp = [colspan_len, rowspan_len - 1]
                    current_line_expands[len(current_line_expands) - 1] = newexp
                    for ci in range(colspan_len - 1):
                        current_line_tags.append("xcel")
                        current_line_expands.append(
                            [colspan_len - ci - 1, rowspan_len - 1]
                        )
                if next_html_tag in pos_colspans:
                    colspan_len = pos_colspans[next_html_tag]
                    tl_spans[current_html_cell_ind - 1] = [colspan_len, rowspan_len]
                    newexp = [colspan_len, rowspan_len - 1]
                    current_line_expands[len(current_line_expands) - 1] = newexp
                    for ci in range(colspan_len - 1):
                        current_line_tags.append("xcel")
                        current_line_expands.append(
                            [colspan_len - ci - 1, rowspan_len - 1]
                        )

    t_name = "*** {}: {} ***".format(table["split"], table["filename"])
    # check if square
    isSquare = otsl_sqr_chk(tl_cells, t_name, logdebug)
    # TODO: pad if not square?
    if not isSquare:
        tl_cells = otsl_pad_to_sqr(tl_cells, "fcel")
    # check if cells (bboxes) in sync:
    if "cells" in out_line["html"]:
        isGood = otsl_tags_cells_sync_chk(
            tl_cells, out_line["html"]["cells"], t_name, logdebug
        )
    # convert back to HTML
    rHTML = []
    if isSquare:
        rHTML = otsl_to_html(tl_cells, logdebug)
    out_line["html"]["html_restored_structure"]["tokens"] = rHTML

    out_line["html"]["structure"]["tokens"] = tl_cells
    out_line["otsl_spans"] = tl_spans
    out_line["cols"] = colnum
    out_line["rows"] = len(t_cells)
    out_line["html_len"] = len(table_html_structure["tokens"])
    out_line["rs_len"] = len(tl_cells)
    # save converted line
    if use_writer:
        if isSquare:
            if isGood:
                writer.write(out_line)

    if logdebug:
        logger.debug("{}Reconstructed HTML:{}".format(bcolors.OKGREEN, bcolors.ENDC))
        logger.debug(rHTML)
        # original HTML
        oHTML = out_line["html"]["html_structure"]
        logger.debug("{}Original HTML:{}".format(bcolors.OKBLUE, bcolors.ENDC))
        logger.debug(oHTML)

    return True, out_line
