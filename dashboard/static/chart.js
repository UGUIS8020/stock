// 累積損益スパークラインのホバー用ツールチップ（外部ライブラリ不使用）
(function () {
  const svg = document.querySelector(".sparkline");
  const tooltip = document.getElementById("spark-tooltip");
  if (!svg || !tooltip) return;

  const dots = svg.querySelectorAll(".spark-dot");
  dots.forEach((dot) => {
    dot.addEventListener("mouseenter", () => {
      dot.classList.add("active");
      const cx = parseFloat(dot.getAttribute("cx"));
      const cy = parseFloat(dot.getAttribute("cy"));
      const pt = svg.createSVGPoint();
      pt.x = cx;
      pt.y = cy;
      const screenPt = pt.matrixTransform(svg.getScreenCTM());
      const rootRect = svg.parentElement.getBoundingClientRect();
      tooltip.textContent = dot.getAttribute("data-value");
      tooltip.style.left = `${screenPt.x - rootRect.left}px`;
      tooltip.style.top = `${screenPt.y - rootRect.top}px`;
      tooltip.hidden = false;
    });
    dot.addEventListener("mouseleave", () => {
      dot.classList.remove("active");
      tooltip.hidden = true;
    });
  });
})();
