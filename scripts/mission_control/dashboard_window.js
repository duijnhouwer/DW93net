// A dedicated viewer: the parent Python process owns and cleans the temporary file.
imports.gi.versions.Gtk = '3.0';
imports.gi.versions.WebKit2 = '4.1';
const {Gtk, WebKit2} = imports.gi;

Gtk.init(null);
const window = new Gtk.Window({title: 'DW93Net — Training progress'});
window.set_default_size(1400, 950);
const view = new WebKit2.WebView();
window.add(view);
window.connect('destroy', () => Gtk.main_quit());
view.load_uri(ARGV[0]);
window.show_all();
Gtk.main();
