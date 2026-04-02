document.addEventListener("DOMContentLoaded", function () {
  var logoLink = document.querySelector(".sphinxsidebar p.logo a");
  var contentRoot = document.documentElement.getAttribute("data-content_root") || "./";

  if (logoLink) {
    logoLink.setAttribute("href", contentRoot + "about.html");
  }

  function fallbackCopyText(text) {
    var textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "absolute";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    fallbackCopyText(text);
    return Promise.resolve();
  }

  var copyButtons = document.querySelectorAll(".install-copy-btn");
  copyButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var targetId = btn.getAttribute("data-copy-target");
      if (!targetId) return;

      var target = document.getElementById(targetId);
      if (!target) return;

      var text = target.textContent.trim();
      copyText(text).then(function () {
        var oldText = btn.textContent;
        btn.textContent = "Copied";
        btn.classList.add("copied");
        setTimeout(function () {
          btn.textContent = oldText;
          btn.classList.remove("copied");
        }, 1200);
      });
    });
  });
});
